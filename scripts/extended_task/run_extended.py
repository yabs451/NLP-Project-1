"""Train the extended-task chain: generation 0 on the real task, then successors.

Generation 0 learns from correct continuations and is shared by every condition.
Each successor learns from continuations its parent generated, mistakes included.
Opening contexts always come from the authors' original task generator.

`--label-strategy` selects how the parent turns label logits into training
targets: argmax, or sampling at `--label-temperature`. The next symbol is always
sampled at temperature 1.

Method and results: findings/03_extended_task_recursion.md

Usage (from the project root):
  .venv/Scripts/python.exe scripts/extended_task/run_extended.py
  .venv/Scripts/python.exe scripts/extended_task/run_extended.py --label-strategy sample --label-temperature 3
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import extended_model as extended

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import main_utils
import samplers

RESULTS = common.ROOT / "results" / "extended_task"
# Generation 0 is trained on real data and is shared by every condition; each
# condition's successors live in their own experiment folder.
GENERATION_0 = RESULTS / "generation_0"
EXPERIMENTS = RESULTS / "experiments"

# Seeds for the parts of the extended task that the base task has no equivalent
# for. Kept distinct from the authors' init (5), train (0) and eval (1) seeds so
# they cannot collide with any existing stream.
SYMBOL_HEAD_INIT_SEED = 11        # initialises the two-way symbol head
TRAINING_CHOICE_SEED = 12         # coin flips for generation 0's next-symbol targets
DEV_CHOICE_SEED = 13              # coin flips for the development set's targets
GENERATION_SEED = 14              # sampling the next symbol when a parent generates
LABEL_SAMPLING_SEED = 15          # sampling labels, when the strategy is "sample"

EVALUATE_EVERY = 5000             # sequences between development evaluations
LARGE_BATCH = 2000                # batch used for evaluation and for generation


def checkpoint_schedule(train_iters, batch_size):
    """The agreed 55 points, rounded up to batch boundaries as training checks them."""
    requested = sorted(set([1000, 2000, 5000, 10000]
                           + list(range(0, train_iters + 1, 20000))))
    return sorted({(value + batch_size - 1) // batch_size * batch_size
                   for value in requested})


def draw_opening_contexts(opts, splits, count):
    """Draw `count` original-task contexts and queries, as compact indices.

    Reuses the authors' samplers, so the opening of every sequence is exactly
    the base task: 50 training classes, exemplar 0, one distractor, training
    label pairs. Returns class indices [n, 3], exemplar indices [n, 3] and the
    true labels [n, 3] (context A, context B, query).
    """
    zipf = jnp.arange(1, opts.class_split[0] + 1, dtype=jnp.float32)
    distribution = (1 / zipf ** opts.zipf_alpha)
    distribution = distribution / jnp.sum(distribution)
    burst_sampler = partial(samplers.get_constant_burst_seq_idxs,
                            classes=splits["class"]["train"], class_distr=distribution,
                            num_seqs=opts.train_bs, context_len=opts.train_context_len,
                            burstiness=1, distractor=1, no_support=0, unique_rest=0)
    # The base task wraps the sampler in get_mixed_seq_idxs even with a single
    # substrate, and that wrapper consumes a key. Keeping it means the openings
    # here are the same questions the original task produces.
    class_sampler = partial(samplers.get_mixed_seq_idxs,
                            mix_probabilities=jnp.array(opts.mixing_coeffs),
                            mix_substrate_fns=[burst_sampler])
    exemplar_sampler = partial(samplers.get_exemplar_inds,
                               allowed_inds=splits["exemplar"]["train"],
                               match_query_and_distractors=opts.match_query_and_distractors)
    train_pairs = splits["relabeling"]["train"]

    @jax.jit
    def one_batch(key):
        keys = jax.random.split(key, 3)
        drawn = class_sampler(keys[0])
        exemplars = exemplar_sampler(keys[1], drawn["idx_types"])
        labels = samplers.fewshot_relabel(keys[2], labels=train_pairs, **drawn)
        return drawn["class_idxs"], exemplars, labels

    # Same key chain the base task uses, so the openings match its question stream.
    data_seed, _ = common.training_seeds(opts)
    classes, exemplars, labels = [], [], []
    for _ in range(count // opts.train_bs):
        data_seed, batch_key = jax.random.split(data_seed)
        one, two, three = one_batch(batch_key)
        classes.append(np.asarray(one, np.int16))
        exemplars.append(np.asarray(two, np.int8))
        labels.append(np.asarray(three, np.int8))
    return np.concatenate(classes), np.concatenate(exemplars), np.concatenate(labels)


def true_continuation_targets(labels, symbol_choice):
    """The correct continuation: the chosen symbol's own context label."""
    return labels[np.arange(len(labels)), symbol_choice]


def build_generation_0_dataset(opts, splits, count):
    """Correct continuations: true query label, a fair coin flip, the true label."""
    class_idxs, exemplar_idxs, labels = draw_opening_contexts(opts, splits, count)
    choice = np.asarray(jax.random.bernoulli(
        jax.random.PRNGKey(TRAINING_CHOICE_SEED), 0.5, (count,)), np.int8)
    next_label = true_continuation_targets(labels, choice)
    return {"class_idxs": class_idxs, "exemplar_idxs": exemplar_idxs,
            "labels": labels, "symbol_choice": choice, "next_label": next_label,
            "true_query_label": labels[:, 2].copy(), "true_next_label": next_label.copy()}


def build_successor_dataset(opts, splits, parent, features, count,
                            label_strategy, label_temperature):
    """Continuations generated by the parent, conditioned on its own earlier tokens.

    The symbol key chain is unchanged from the argmax condition; label sampling
    draws from its own separate chain, so switching strategy cannot disturb the
    opening questions or the symbol stream.
    """
    class_idxs, exemplar_idxs, labels = draw_opening_contexts(opts, splits, count)
    true_query = labels[:, 2].copy()

    generated_query = np.empty(count, np.int8)
    generated_choice = np.empty(count, np.int8)
    generated_next = np.empty(count, np.int8)
    key = jax.random.PRNGKey(GENERATION_SEED)
    label_key = jax.random.PRNGKey(LABEL_SAMPLING_SEED)
    for start in range(0, count, LARGE_BATCH):
        stop = start + LARGE_BATCH
        key, step_key = jax.random.split(key)
        label_key, label_step_key = jax.random.split(label_key)
        symbols = features[jnp.asarray(class_idxs[start:stop]),
                           jnp.asarray(exemplar_idxs[start:stop])]
        query, choice, following = extended.generate_continuation(
            parent, symbols, jnp.asarray(labels[start:stop, :2], jnp.int32), step_key,
            label_step_key, label_strategy, label_temperature)
        generated_query[start:stop] = np.asarray(query, np.int8)
        generated_choice[start:stop] = np.asarray(choice, np.int8)
        generated_next[start:stop] = np.asarray(following, np.int8)

    # The successor trains on the parent's targets; the true answers are kept
    # only so the generated continuations can be scored afterwards.
    stored = labels.copy()
    stored[:, 2] = generated_query
    return {"class_idxs": class_idxs, "exemplar_idxs": exemplar_idxs,
            "labels": stored, "symbol_choice": generated_choice,
            "next_label": generated_next, "true_query_label": true_query,
            "true_next_label": true_continuation_targets(labels, generated_choice)}


def dataset_quality(dataset):
    """How closely a generated dataset matches the correct continuations."""
    choice = dataset["symbol_choice"]
    query_errors = int(np.sum(dataset["labels"][:, 2] != dataset["true_query_label"]))
    next_errors = int(np.sum(dataset["next_label"] != dataset["true_next_label"]))
    return {
        "examples": int(len(choice)),
        "query_label_errors": query_errors,
        "query_label_error_rate": query_errors / len(choice),
        "next_label_errors": next_errors,
        "next_label_error_rate": next_errors / len(choice),
        "chose_context_position_0": float(np.mean(choice == 0)),
        "distinct_next_symbol_classes": int(len(np.unique(
            dataset["class_idxs"][np.arange(len(choice)), choice]))),
        "next_symbol_class_counts_top5": np.bincount(
            dataset["class_idxs"][np.arange(len(choice)), choice]).argsort()[-5:][::-1].tolist(),
    }


def load_dev_set(opts):
    """The fixed development prefixes, extended with reproducible continuation targets.

    The 1,000 opening questions and their held-out classes are unchanged. The
    next-symbol target is a fair coin flip from a fixed seed, so the extension is
    identical for every generation.
    """
    with h5py.File(common.DEV_EVALUATOR_FILE, "r") as handle:
        name = list(handle.keys())[0]
        symbols = jnp.asarray(handle[name]["examples"][:])
        labels = np.asarray(handle[name]["labels"][:])
    choice = np.asarray(jax.random.bernoulli(
        jax.random.PRNGKey(DEV_CHOICE_SEED), 0.5, (len(labels),)), np.int32)
    next_label = true_continuation_targets(labels, choice)
    return {"symbols": symbols, "labels": jnp.asarray(labels, jnp.int32),
            "symbol_choice": jnp.asarray(choice), "next_label": jnp.asarray(next_label),
            "true_next_label": jnp.asarray(next_label)}


def score_dev(model, dev, key):
    """Teacher-forced development scores, averaged over batches."""
    totals = []
    for start in range(0, len(dev["symbol_choice"]), LARGE_BATCH):
        stop = start + LARGE_BATCH
        totals.append(extended.evaluate_teacher_forced(
            model, dev["symbols"][start:stop], dev["labels"][start:stop],
            dev["symbol_choice"][start:stop], dev["next_label"][start:stop],
            dev["true_next_label"][start:stop], key))
    return {metric: float(np.mean([float(batch[metric]) for batch in totals]))
            for metric in totals[0]}


def train_generation(opts, folder, dataset, features, dev):
    """One pass over the dataset, saving the 55 checkpoints and the metric log."""
    model = extended.build_model(opts, SYMBOL_HEAD_INIT_SEED)
    optimizer = main_utils.get_optimizer_from_opts(opts)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    _, train_model_seed = common.training_seeds(opts)

    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    schedule = set(checkpoint_schedule(opts.train_iters, opts.train_bs))

    class_idxs = jnp.asarray(dataset["class_idxs"], jnp.int32)
    exemplar_idxs = jnp.asarray(dataset["exemplar_idxs"], jnp.int32)
    labels = jnp.asarray(dataset["labels"], jnp.int32)
    choice = jnp.asarray(dataset["symbol_choice"], jnp.int32)
    next_label = jnp.asarray(dataset["next_label"], jnp.int32)

    log = {name: [] for name in ("train_total", "train_query", "train_symbol",
                                 "train_next", "train_iter")}
    dev_log, eval_iters = [], []

    def save(sequences):
        eqx.tree_serialise_leaves(checkpoints / "{:011d}.eqx".format(sequences), model)

    for sequences in range(0, opts.train_iters, opts.train_bs):
        if sequences in schedule:
            save(sequences)
        if sequences % EVALUATE_EVERY < opts.train_bs:
            dev_log.append(score_dev(model, dev, jax.random.PRNGKey(0)))
            eval_iters.append(sequences)

        stop = sequences + opts.train_bs
        symbols = features[class_idxs[sequences:stop], exemplar_idxs[sequences:stop]]
        train_model_seed, step_key = jax.random.split(train_model_seed)
        model, opt_state, total, parts = extended.train_step(
            model, optimizer, opt_state, symbols, labels[sequences:stop],
            choice[sequences:stop], next_label[sequences:stop], step_key)
        log["train_total"].append(float(total))
        log["train_query"].append(float(parts[0]))
        log["train_symbol"].append(float(parts[1]))
        log["train_next"].append(float(parts[2]))
        log["train_iter"].append(stop)

    save(opts.train_iters)
    dev_log.append(score_dev(model, dev, jax.random.PRNGKey(0)))
    eval_iters.append(opts.train_iters)

    with h5py.File(folder / "log.h5", "w") as handle:
        for name, values in log.items():
            handle.create_dataset(name, data=np.asarray(values, np.float32))
        handle.create_dataset("eval_iter", data=np.asarray(eval_iters, np.int64))
        for metric in dev_log[0]:
            handle.create_dataset("dev/" + metric,
                                  data=np.asarray([row[metric] for row in dev_log], np.float32))
    assert np.isfinite(log["train_total"]).all(), "non-finite training loss"
    return model


def save_dataset(path, dataset):
    with h5py.File(path, "w") as handle:
        for name, array in dataset.items():
            handle.create_dataset(name, data=array, compression="gzip")


def write_config(folder, opts, generation, parent_folder, label_strategy,
                 label_temperature):
    """One record per run: the settings, the seeds and where the parent came from."""
    config = {key: value for key, value in vars(opts).items()
              if not key.startswith("opto_")}
    config.update(
        run=folder.name, generation=generation, task="extended (query label, next symbol, its label)",
        train_seed=int(opts.train_seed), eval_seed=int(opts.eval_seed),
        init_seed=int(opts.init_seed),
        parent_run=(None if parent_folder is None
                    else str(Path(parent_folder).relative_to(common.ROOT))),
        symbol_head_init_seed=SYMBOL_HEAD_INIT_SEED,
        training_choice_seed=TRAINING_CHOICE_SEED,
        dev_choice_seed=DEV_CHOICE_SEED,
        generation_sampling_seed=GENERATION_SEED,
        label_sampling_seed=LABEL_SAMPLING_SEED,
        next_symbol_rule="sampled from the two-way head at temperature 1",
        label_strategy=label_strategy,
        label_temperature=(label_temperature if label_strategy == "sample" else None),
        label_rule=("argmax over the five labels" if label_strategy == "argmax" else
                    "sampled from softmax(logits / {}) over all five labels".format(
                        label_temperature)),
        dev_evaluator=str(common.DEV_EVALUATOR_FILE.relative_to(common.ROOT)))
    (folder / "config.json").write_bytes(
        json.dumps(config, indent=2, default=str).encode("utf-8"))


def condition_name(label_strategy, label_temperature):
    """Folder name for a condition, e.g. label_argmax or label_sampling_temperature_3."""
    if label_strategy == "argmax":
        return "label_argmax"
    return "label_sampling_temperature_{:g}".format(label_temperature)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=4,
                        help="How many successors to train after generation 0")
    parser.add_argument("--label-strategy", choices=["argmax", "sample"], default="argmax",
                        help="How the parent turns label logits into training targets")
    parser.add_argument("--label-temperature", type=float, default=3.0,
                        help="Temperature used when --label-strategy is 'sample'")
    args = parser.parse_args()
    common.use_above_normal_priority()

    # The authors' published settings, at their original learning rate. Nothing
    # is tuned for the extended task, and nothing but label generation differs
    # between conditions.
    opts = common.baseline_options()
    opts.model_output_classes = opts.fs_relabel
    features = common.load_features()
    splits = main_utils.get_splits_from_opts(opts, features.shape)
    dev = load_dev_set(opts)

    condition = EXPERIMENTS / condition_name(args.label_strategy, args.label_temperature)
    condition.mkdir(parents=True, exist_ok=True)
    print("condition:", condition.relative_to(common.ROOT), flush=True)

    # Generation 0 is shared: train it once, then reuse it for every condition.
    if (GENERATION_0 / "log.h5").exists():
        parent_model = load_final_model(GENERATION_0, opts)
    else:
        GENERATION_0.mkdir(parents=True, exist_ok=True)
        dataset = build_generation_0_dataset(opts, splits, opts.train_iters)
        save_dataset(GENERATION_0 / "training_data.h5", dataset)
        (GENERATION_0 / "dataset_quality.json").write_bytes(
            json.dumps(dataset_quality(dataset), indent=2).encode("utf-8"))
        write_config(GENERATION_0, opts, 0, None, "argmax", None)
        parent_model = train_generation(opts, GENERATION_0, dataset, features, dev)
        print("generation 0 trained", flush=True)
    parent_folder = GENERATION_0

    for generation in range(1, args.generations + 1):
        folder = condition / "generation_{}".format(generation)
        if (folder / "log.h5").exists():
            print("generation", generation, "already trained; loading it", flush=True)
            parent_model = load_final_model(folder, opts)
            parent_folder = folder
            continue
        folder.mkdir(parents=True, exist_ok=True)

        dataset = build_successor_dataset(opts, splits, parent_model, features,
                                          opts.train_iters, args.label_strategy,
                                          args.label_temperature)
        save_dataset(folder / "training_data.h5", dataset)
        quality = dataset_quality(dataset)
        (folder / "dataset_quality.json").write_bytes(
            json.dumps(quality, indent=2).encode("utf-8"))
        print("generation", generation, "dataset:", json.dumps(quality), flush=True)

        write_config(folder, opts, generation, parent_folder, args.label_strategy,
                     args.label_temperature)
        parent_model = train_generation(opts, folder, dataset, features, dev)
        parent_folder = folder
        print("generation", generation, "trained", flush=True)

    print("done;", condition.relative_to(common.ROOT), "holds generations 1 to",
          args.generations)


def load_model(folder, sequences, opts):
    """Deserialise one saved checkpoint of a generation."""
    template = extended.build_model(opts, SYMBOL_HEAD_INIT_SEED)
    path = Path(folder) / "checkpoints" / "{:011d}.eqx".format(sequences)
    return eqx.tree_deserialise_leaves(path, template)


def load_final_model(folder, opts):
    """Deserialise a finished generation's final model."""
    last = max(int(p.stem) for p in (Path(folder) / "checkpoints").glob("*.eqx"))
    return load_model(folder, last, opts)


if __name__ == "__main__":
    main()
