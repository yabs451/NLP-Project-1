"""Recursive training for the base (label-only) task.

One generation of the loop does three things:

  A. Draw training questions from the *original* training distribution --
     the same 50 training classes, exemplar 0, context construction and
     training label pairs the authors' baseline uses. Context labels stay
     correct; only the query's answer will be replaced.
  B. Ask the parent model for the answer to each question and keep its
     argmax over all five output labels as the new training target.
  C. Train a freshly initialised successor on that saved dataset for exactly
     one pass, then score it on the same fixed development evaluator the
     parent used.

Generation 0 is the original model trained on the real task. Running this
script once with --generations 1 produces generation 1. Pointing --parent at
generation 1 later produces generation 2, with no code changes.

Usage (from the project root):
  .venv/Scripts/python.exe scripts/base_task/run_recursive.py \
      --parent results/base_task/generation_0_original_data_init_seed_5 \
      --generations 1
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # sets JAX_PLATFORMS and puts the authors' code on sys.path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import main as upstream_main
import main_utils
import opto
import samplers

# How many sequences the parent labels at once. Large enough to keep JAX busy,
# small enough that activations for the batch stay comfortably in memory.
INFERENCE_BATCH = 2000
RESULTS = common.ROOT / "results" / "base_task"


def generation_number(run_folder):
    """Read a run's generation index; runs without the file are generation 0."""
    record = Path(run_folder) / "generation_metadata.json"
    if record.exists():
        return int(json.loads(record.read_text())["generation"])
    return 0


def question_sampler(opts, splits):
    """A jitted sampler returning the *indices* behind one batch of questions.

    The authors' `make_data_sampler` returns the 512-dimensional feature
    vectors directly. We need the class and exemplar indices instead, so the
    saved dataset can reference the feature file rather than copying it. This
    reproduces their `sample` closure (samplers.py:308-330) key split for key
    split, so the questions are the same ones their sampler would produce.

    Returns class_idxs [batch, 3], exemplar_idxs [batch, 3], labels [batch, 3].
    """
    zipf_ranks = jnp.arange(1, opts.class_split[0] + 1, dtype=jnp.float32)
    zipf_probs = 1 / zipf_ranks ** opts.zipf_alpha
    class_distr = zipf_probs / jnp.sum(zipf_probs)

    # The baseline uses exactly one sampler with mixing weight 1.0.
    assert opts.mixing_coeffs == [1.0], "recursive pipeline assumes the single baseline sampler"
    burst_sampler = partial(samplers.get_constant_burst_seq_idxs,
                            classes=splits["class"]["train"], class_distr=class_distr,
                            num_seqs=opts.train_bs, context_len=opts.train_context_len,
                            burstiness=opts.pt_burstiness[0],
                            distractor=upstream_main.smart_index(opts.pt_distract, 0, 1),
                            no_support=upstream_main.smart_index(opts.pt_no_support, 0, 0),
                            unique_rest=upstream_main.smart_index(opts.pt_unique_rest, 0, 0))
    class_sampler = partial(samplers.get_mixed_seq_idxs,
                            mix_probabilities=jnp.array(opts.mixing_coeffs),
                            mix_substrate_fns=[burst_sampler])
    exemplar_sampler = partial(samplers.get_exemplar_inds,
                               allowed_inds=splits["exemplar"]["train"],
                               match_query_and_distractors=opts.match_query_and_distractors)
    train_pairs = splits["relabeling"]["train"]

    @jax.jit
    def sample(key):
        keys = jax.random.split(key, 3)
        class_out = class_sampler(keys[0])
        exemplar_idxs = exemplar_sampler(keys[1], class_out["idx_types"])
        labels = samplers.fewshot_relabel(keys[2], labels=train_pairs, **class_out)
        return class_out["class_idxs"], exemplar_idxs, labels

    return sample


def parent_predictor(opts, model):
    """Batched argmax prediction from the parent, with no weight updates.

    The query label is never part of the model input: the authors'
    SequenceClassifier drops it when interleaving tokens (models.py:752). We
    still pass the true labels array, exactly as their evaluation path does,
    and verify the non-dependence separately in `check_no_target_leak`.
    """
    forward = opto.make_fn_from_opts(opts)
    assert forward is opto.default_model_fwd_fn, "parent must run without interventions"

    @eqx.filter_jit
    def predict(x, y, key):
        keys = jax.random.split(key, x.shape[0])
        out = jax.vmap(partial(forward, model=model))(x=x, y=y, key=keys)
        return out["out"][:, -1, :]      # logits at the query position, [batch, 5]

    return predict


def check_no_target_leak(predict, features, class_idxs, exemplar_idxs, labels, num_classes):
    """Changing the query's target must not change the parent's prediction."""
    x = features[class_idxs[:64], exemplar_idxs[:64]]
    y = jnp.asarray(labels[:64])
    key = jax.random.PRNGKey(0)
    original = predict(x, y, key)
    altered = predict(x, y.at[:, -1].set((y[:, -1] + 1) % num_classes), key)
    assert np.array_equal(np.asarray(original), np.asarray(altered)), \
        "query target reached the model input"


def generate_dataset(opts, parent_model, features, splits, output_path):
    """Build and save one generation's training data. Returns quality statistics.

    The question stream reuses the baseline's own training key chain
    (main.py:347,525), so the successor sees the same questions in the same
    order as its parent did -- only the query answers differ.
    """
    sample = question_sampler(opts, splits)
    predict = parent_predictor(opts, parent_model)
    num_batches = opts.train_iters // opts.train_bs

    # Stage 1: draw every question, keeping only indices and labels.
    train_data_seed, _ = common.training_seeds(opts)
    all_classes, all_exemplars, all_labels = [], [], []
    for _ in range(num_batches):
        train_data_seed, batch_seed = jax.random.split(train_data_seed)
        class_idxs, exemplar_idxs, labels = sample(batch_seed)
        all_classes.append(np.asarray(class_idxs, dtype=np.int16))
        all_exemplars.append(np.asarray(exemplar_idxs, dtype=np.int8))
        all_labels.append(np.asarray(labels, dtype=np.int8))
    class_idxs = np.concatenate(all_classes)
    exemplar_idxs = np.concatenate(all_exemplars)
    labels = np.concatenate(all_labels)
    true_query_label = labels[:, -1].copy()
    assert len(class_idxs) == opts.train_iters

    check_no_target_leak(predict, features, class_idxs, exemplar_idxs, labels,
                         opts.model_output_classes)

    # Stage 2: label every question with the parent's argmax over all 5 labels.
    generated = np.empty(len(class_idxs), dtype=np.int8)
    for start in range(0, len(class_idxs), INFERENCE_BATCH):
        stop = start + INFERENCE_BATCH
        x = features[class_idxs[start:stop], exemplar_idxs[start:stop]]
        y = jnp.asarray(labels[start:stop])
        logits = predict(x, y, jax.random.PRNGKey(0))
        generated[start:stop] = np.asarray(jnp.argmax(logits, axis=-1), dtype=np.int8)

    # Replace ONLY the query target; context labels and symbols are untouched.
    labels[:, -1] = generated
    assert np.array_equal(labels[:, :-1], np.concatenate(all_labels)[:, :-1])

    in_context = (generated == labels[:, 0]) | (generated == labels[:, 1])
    statistics = {
        "examples": int(len(generated)),
        "parent_argmax_error_rate_vs_true_answer": float(np.mean(generated != true_query_label)),
        "parent_out_of_context_label_rate": float(np.mean(~in_context)),
        "generated_label_distribution": np.bincount(
            generated, minlength=opts.model_output_classes).tolist(),
        "true_label_distribution": np.bincount(
            true_query_label, minlength=opts.model_output_classes).tolist(),
        "distinct_questions": int(len(np.unique(
            np.concatenate([class_idxs, exemplar_idxs,
                            np.concatenate(all_labels)[:, :-1]], axis=1), axis=0))),
    }

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("class_idxs", data=class_idxs, compression="gzip")
        handle.create_dataset("exemplar_idxs", data=exemplar_idxs, compression="gzip")
        handle.create_dataset("labels", data=labels, compression="gzip")
        # Kept for diagnostics only. Never used as a training target here.
        handle.create_dataset("true_query_label", data=true_query_label, compression="gzip")
    return statistics, predict, (class_idxs, exemplar_idxs, labels, true_query_label)


def worked_example(predict, features, class_idxs, exemplar_idxs, labels,
                   true_query_label, index):
    """Write one real sequence out in full, with the parent's probabilities."""
    index = int(index)
    x = features[class_idxs[index:index + 1], exemplar_idxs[index:index + 1]]
    y = jnp.asarray(labels[index:index + 1])
    probabilities = jax.nn.softmax(predict(x, y, jax.random.PRNGKey(0))[0])
    return {
        "dataset_index": index,
        "context": [{"symbol_class": int(class_idxs[index][slot]),
                     "exemplar": int(exemplar_idxs[index][slot]),
                     "label": int(labels[index][slot])} for slot in range(2)],
        "query_symbol_class": int(class_idxs[index][2]),
        "query_exemplar": int(exemplar_idxs[index][2]),
        "true_answer": int(true_query_label[index]),
        "parent_probabilities_over_5_labels": [round(float(p), 6) for p in probabilities],
        "parent_argmax_target_used_for_training": int(labels[index][2]),
        "target_matches_true_answer": bool(labels[index][2] == true_query_label[index]),
    }


def worked_examples(predict, features, class_idxs, exemplar_idxs, labels, true_query_label):
    """A typical sequence, plus a genuinely mislabelled one if any exists.

    Recursive training only degrades a model through targets that are wrong, so
    the errors are the interesting cases. If the parent made no mistakes we say
    so rather than inventing one.
    """
    examples = {"typical": worked_example(predict, features, class_idxs, exemplar_idxs,
                                          labels, true_query_label, 0)}
    wrong = np.where(labels[:, -1] != true_query_label)[0]
    if len(wrong) == 0:
        examples["mislabelled"] = None
        examples["note"] = "the parent's argmax matched the true answer on every question"
        return examples
    examples["mislabelled"] = worked_example(predict, features, class_idxs, exemplar_idxs,
                                             labels, true_query_label, wrong[0])
    # A deterministic parent mislabels a given question every time it appears,
    # so count distinct questions rather than just occurrences.
    questions = np.concatenate([class_idxs, exemplar_idxs, labels[:, :-1]], axis=1)
    distinct_wrong = np.unique(questions[wrong], axis=0)
    examples["mislabelled_occurrences"] = int(len(wrong))
    examples["distinct_mislabelled_questions"] = int(len(distinct_wrong))
    return examples


def train_successor(opts, folder, features, dataset, evaluators, log_path):
    """One pass over the saved dataset, reusing the authors' update and evaluation.

    Mirrors the structure of main.py's training loop (main.py:468-545) but reads
    batches from the saved dataset instead of sampling them. The model starts
    from a fresh initialisation (init_seed), not from the parent's weights.
    """
    class_idxs, exemplar_idxs, labels, _ = dataset
    model, optimizer = common.fresh_model_and_optimizer(opts)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    forward = opto.make_fn_from_opts(opts)
    _, train_model_seed = common.training_seeds(opts)
    _, eval_model_seed = common.evaluation_seeds(opts)

    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    results = h5py.File(log_path, "a")
    for metric in upstream_main.ALL_TRAIN_METRICS + ["iter"]:
        results.create_dataset("train_" + metric, shape=(0,), maxshape=(None,), dtype=float)
    results.create_dataset("eval_iter", shape=(0,), maxshape=(None,), dtype=int)
    results.close()

    def run_evaluation(model, eval_model_seed, sequences, pending):
        eval_model_seed, key = jax.random.split(eval_model_seed, 2)
        out = upstream_main.evaluate(model=model, fwd_fn=forward, key=key,
                                     eval_data=evaluators, eval_batch_size=opts.eval_bs)
        with h5py.File(log_path, "a") as handle:
            for name in out:
                for metric in out[name]:
                    address = "/".join([name, metric])
                    if address in handle:
                        handle[address].resize(handle[address].shape[0] + 1, axis=0)
                        handle[address][-1] = out[name][metric]
                    else:
                        handle.create_dataset(address, data=jnp.broadcast_to(
                            out[name][metric], (1, len(out[name][metric]))),
                            maxshape=(None, len(out[name][metric])))
            handle["eval_iter"].resize(handle["eval_iter"].shape[0] + 1, axis=0)
            handle["eval_iter"][-1] = sequences
            for metric, values in pending.items():
                if values:
                    key_name = "train_" + metric
                    handle[key_name].resize(handle[key_name].shape[0] + len(values), axis=0)
                    handle[key_name][-len(values):] = values
        print("-" * 10 + str(sequences), "dev acc",
              float(jnp.mean(out["fsl_dev_class"]["acc"])), flush=True)
        return eval_model_seed, {m: [] for m in pending}

    def save_checkpoint(model, opt_state, sequences, eval_model_seed, train_model_seed):
        eqx.tree_serialise_leaves(
            checkpoints / "{:011d}.eqx".format(sequences),
            {"iter": sequences, "model": model, "opt_state": opt_state,
             "seeds": {"eval_model_seed": eval_model_seed,
                       "train_data_seed": train_model_seed,
                       "train_model_seed": train_model_seed}})

    pending = {metric: [] for metric in upstream_main.ALL_TRAIN_METRICS + ["iter"]}
    eval_index = ckpt_index = 0
    for sequences in range(0, opts.train_iters, opts.train_bs):
        if eval_index < len(opts.eval_sched) and sequences >= opts.eval_sched[eval_index]:
            eval_model_seed, pending = run_evaluation(model, eval_model_seed, sequences, pending)
            eval_index += 1
        if ckpt_index < len(opts.ckpt_sched) and sequences >= opts.ckpt_sched[ckpt_index]:
            save_checkpoint(model, opt_state, sequences, eval_model_seed, train_model_seed)
            ckpt_index += 1

        stop = sequences + opts.train_bs
        x = features[class_idxs[sequences:stop], exemplar_idxs[sequences:stop]]
        y = jnp.asarray(labels[sequences:stop])
        train_model_seed, step_seed = jax.random.split(train_model_seed)
        metrics, model, opt_state = upstream_main.train_step(
            model=model, fwd_fn=forward, optimizer=optimizer, opt_state=opt_state,
            microbs=opts.train_microbs, weight_decay=opts.weight_decay,
            x=x, y=y, key=step_seed)
        pending["iter"].append(stop)
        for metric in metrics:
            pending[metric].append(metrics[metric])

    eval_model_seed, pending = run_evaluation(model, eval_model_seed, opts.train_iters, pending)
    save_checkpoint(model, opt_state, opts.train_iters, eval_model_seed, train_model_seed)
    return int(opt_state[0].count)


def run_one_generation(parent_folder, opts, features, splits, evaluators):
    """Generate data from the parent, then train and record one successor."""
    number = generation_number(parent_folder) + 1
    folder = RESULTS / "generation_{}_generated_data_init_seed_{}".format(number, opts.init_seed)
    if folder.exists():
        raise SystemExit("{} already exists; refusing to overwrite".format(folder))
    folder.mkdir(parents=True)

    parent_iters = common.available_checkpoints(parent_folder)[-1]
    parent = common.load_checkpoint(parent_folder, parent_iters, opts)
    print("parent:", parent_folder, "checkpoint", parent_iters, flush=True)

    dataset_path = folder / "generated_training_data.h5"
    statistics, predict, dataset = generate_dataset(
        opts, parent["model"], features, splits, dataset_path)
    class_idxs, exemplar_idxs, labels, true_query_label = dataset

    examples = worked_examples(predict, features, class_idxs, exemplar_idxs,
                               labels, true_query_label)

    metadata = {
        "generation": number,
        "task": "base (label-only): the successor predicts the query label",
        "parent_run": str(Path(parent_folder).relative_to(common.ROOT)),
        "parent_checkpoint": str(common.checkpoint_path(parent_folder, parent_iters)
                                 .relative_to(common.ROOT)),
        "parent_checkpoint_sequences": int(parent_iters),
        "source_features": str(common.FEATURE_FILE.relative_to(common.ROOT)),
        "generation_rule": ("questions drawn from the original training distribution "
                            "(50 training classes, exemplar 0, burstiness 1, one distractor, "
                            "training label pairs); the query target is replaced by the parent's "
                            "argmax over all five output labels; context labels and symbols kept"),
        "question_stream": ("reuses the baseline's own training key chain from train_seed, so the "
                            "questions and their order match the parent's training stream"),
        "dataset_file": dataset_path.name,
        "target_quality": statistics,
        "worked_examples": examples,
    }
    (folder / "generation_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"target_quality": statistics, "worked_examples": examples}, indent=2), flush=True)

    # The successor's config is the parent's, with our own bookkeeping added, so
    # that analyse_runs.py and the authors' tooling read it unchanged.
    config = dict(vars(opts))
    config["train_seed"] = int(opts.train_seed)
    config["eval_seed"] = int(opts.eval_seed)
    config["run"] = folder.name
    config["base_folder"] = str(RESULTS)
    config["load_eval_data"] = [str(common.DEV_EVALUATOR_FILE)]
    (folder / "config.json").write_text(json.dumps(config, indent=2, default=str))

    updates = train_successor(opts, folder, features, dataset, evaluators, folder / "log.h5")
    assert updates == opts.train_iters // opts.train_bs, "unexpected optimizer update count"
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", type=Path, required=True,
                        help="Run folder to start from (generation 0 is the original baseline)")
    parser.add_argument("--generations", type=int, default=1,
                        help="How many additional generations to train, one after another")
    args = parser.parse_args()
    args.parent = args.parent.resolve()

    # Every generation inherits the parent's scientific settings unchanged, so
    # nothing but the training data differs between generations.
    opts = common.load_run_options(args.parent)
    opts.load_eval_data = [str(common.DEV_EVALUATOR_FILE)]
    features = common.load_features()
    splits = main_utils.get_splits_from_opts(opts, features.shape)
    evaluators = common.build_evaluators(opts, features)
    print("evaluators:", sorted(evaluators), flush=True)

    parent = args.parent
    for _ in range(args.generations):
        parent = run_one_generation(parent, opts, features, splits, evaluators)
        print("finished:", parent, flush=True)
    print("To continue, pass --parent", parent.relative_to(common.ROOT), flush=True)


if __name__ == "__main__":
    main()
