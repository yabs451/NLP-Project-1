"""Train the extended-task chain: generation 0 on the real task, then successors.

Generation 0 learns from correct continuations and is shared by every condition.
Each successor learns from continuations its parent generated, mistakes included.
Opening contexts always come from the authors' original task generator.

Three things can vary, one family at a time, from the reference condition
(original questions, argmax labels, next symbol at temperature 1):
`--label-strategy`/`--label-temperature` change how label targets are chosen,
`--next-symbol-temperature` how sharply the parent picks between the two context
symbols, and `--context-temperature` turns on context feedback, where each
child's questions are built from the parent's own generated symbol frequencies.
The learning rate comes from the extended-task search.

Method and results: findings/05_label_generation_strategies.md and
findings/06_symbol_distribution_experiments.md

Usage (from the project root), one command per condition:
  .venv/Scripts/python.exe scripts/extended_task/run_extended.py
  .venv/Scripts/python.exe scripts/extended_task/run_extended.py --next-symbol-temperature 1/3
  .venv/Scripts/python.exe scripts/extended_task/run_extended.py --context-temperature 1/3
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
TUNING = RESULTS / "tuning"
SELECTION_FILE = TUNING / "selection.json"
RECURSIVE = RESULTS / "recursive"
# Generation 0 is trained on real data and is shared by every condition; each
# condition's successors live in their own experiment folder.
GENERATION_0 = RECURSIVE / "generation_0"
EXPERIMENTS = RECURSIVE / "experiments"

# Seeds for the parts of the extended task that the base task has no equivalent
# for. Kept distinct from the authors' init (5), train (0) and eval (1) seeds so
# they cannot collide with any existing stream.
SYMBOL_HEAD_INIT_SEED = 11        # initialises the two-way symbol head
TRAINING_CHOICE_SEED = 12         # coin flips for generation 0's next-symbol targets
DEV_CHOICE_SEED = 13              # coin flips for the development set's targets
GENERATION_SEED = 14              # sampling the next symbol when a parent generates
LABEL_SAMPLING_SEED = 15          # sampling labels, when the strategy is "sample"
FREQUENCY_SEED = 16               # measuring a parent's own generated symbol frequencies

EVALUATE_EVERY = 5000             # sequences between development evaluations
LARGE_BATCH = 2000                # batch used for evaluation and for generation


def checkpoint_schedule(train_iters, batch_size):
    """The agreed 55 points, rounded up to batch boundaries as training checks them."""
    requested = sorted(set([1000, 2000, 5000, 10000]
                           + list(range(0, train_iters + 1, 20000))))
    return sorted({(value + batch_size - 1) // batch_size * batch_size
                   for value in requested})


def selected_learning_rate():
    """The rate chosen by the extended-task search, read from its selection record.

    Reading it rather than hard-coding it means the recursive chain cannot drift
    from the search that justified it.
    """
    if not SELECTION_FILE.exists():
        raise SystemExit("Missing {}. Run scripts/extended_task/tune_extended.py "
                         "first.".format(SELECTION_FILE.relative_to(common.ROOT)))
    return float(json.loads(SELECTION_FILE.read_text())["selected_learning_rate"])


def draw_opening_contexts(opts, splits, count, class_weights=None):
    """Draw `count` original-task contexts and queries, as compact indices.

    Reuses the authors' samplers, so the opening of every sequence is exactly
    the base task: 50 training classes, exemplar 0, one distractor, training
    label pairs. Returns class indices [n, 3], exemplar indices [n, 3] and the
    true labels [n, 3] (context A, context B, query).

    `class_weights` replaces the original uniform class distribution, in the
    order of `splits["class"]["train"]`. The sampler draws the query class from
    it and then the distractor from the same weights with the query masked out,
    so the two context classes stay distinct and the offered frequencies need
    not match the weights exactly. Everything else — label pairs, query
    construction, exemplars — is unchanged.
    """
    if class_weights is None:
        zipf = jnp.arange(1, opts.class_split[0] + 1, dtype=jnp.float32)
        distribution = (1 / zipf ** opts.zipf_alpha)
    else:
        distribution = jnp.asarray(class_weights, jnp.float32)
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


def measure_generated_symbol_frequencies(opts, parent, features, dataset, train_classes,
                                        symbol_temperature):
    """Which symbol identities the parent generates on its own training questions.

    Step (a) of the context-feedback rule: replay the openings already saved in
    the parent's dataset, generate continuations the ordinary way, and count the
    classes chosen. Counts are accumulated batch by batch so no second
    million-example dataset is written. Returns frequencies in the order of
    `train_classes`.

    These are the identities the parent *produced*, not the ones it inherited in
    its own targets.
    """
    class_idxs = dataset["class_idxs"]
    labels = dataset["labels"]
    counts = np.zeros(int(train_classes.max()) + 1, np.int64)
    key = jax.random.PRNGKey(FREQUENCY_SEED)
    label_key = jax.random.PRNGKey(LABEL_SAMPLING_SEED)
    for start in range(0, len(class_idxs), LARGE_BATCH):
        stop = start + LARGE_BATCH
        key, step_key = jax.random.split(key)
        label_key, label_step_key = jax.random.split(label_key)
        symbols = features[jnp.asarray(class_idxs[start:stop]),
                           jnp.asarray(dataset["exemplar_idxs"][start:stop])]
        _, choice, _ = extended.generate_continuation(
            parent, symbols, jnp.asarray(labels[start:stop, :2], jnp.int32), step_key,
            label_step_key, "argmax", 1.0, symbol_temperature)
        chosen = class_idxs[start:stop][np.arange(stop - start), np.asarray(choice)]
        counts += np.bincount(chosen.astype(np.int64), minlength=len(counts))
    frequencies = counts[train_classes] / counts.sum()
    return frequencies


def context_sampling_weights(frequencies, context_temperature):
    """Sharpen measured frequencies into class weights: w_i proportional to p_i ** (1/T).

    Temperature 1 leaves them unchanged, 1/3 cubes them and 0.2 raises them to
    the fifth power. The exponent is applied to the frequencies themselves, not
    to raw counts through a softmax. No smoothing or cutoff is applied, so a
    class the parent never generated keeps weight zero.
    """
    weights = np.asarray(frequencies, np.float64) ** (1.0 / context_temperature)
    total = weights.sum()
    if np.count_nonzero(weights) < 2 or total <= 0:
        raise SystemExit("context feedback left fewer than two classes with positive "
                         "weight, so valid two-class questions can no longer be built")
    return weights / total


def build_successor_dataset(opts, splits, parent, features, count,
                            label_strategy, label_temperature,
                            symbol_temperature=1.0, class_weights=None):
    """Continuations generated by the parent, conditioned on its own earlier tokens.

    The symbol key chain is unchanged from the argmax condition; label sampling
    draws from its own separate chain, so switching strategy cannot disturb the
    opening questions or the symbol stream. `class_weights` changes which
    symbols the questions are built from, and `symbol_temperature` how sharply
    the parent picks between the two in each context; both change what is
    sampled, so the outputs are not expected to match the reference condition.
    """
    class_idxs, exemplar_idxs, labels = draw_opening_contexts(opts, splits, count,
                                                              class_weights)
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
            label_step_key, label_strategy, label_temperature, symbol_temperature)
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
    """How closely a generated dataset matches the correct continuations.

    Errors are reported as counts as well as rates, because at low error rates
    rounding hides a real and growing number of wrong targets.

    Two different things are measured about the next symbol and kept apart:
    *which context position* the parent preferred, and *which symbol identities*
    ended up in the data. The identity spread is summarised by how many classes
    appear, the largest single class share, and the Shannon entropy of the class
    distribution in nats; 50 evenly used training classes would give ln 50 =
    3.912.
    """
    choice = dataset["symbol_choice"]
    chosen_class = dataset["class_idxs"][np.arange(len(choice)), choice]
    counts = np.bincount(chosen_class)
    present = counts[counts > 0] / len(choice)
    query_errors = int(np.sum(dataset["labels"][:, 2] != dataset["true_query_label"]))
    next_errors = int(np.sum(dataset["next_label"] != dataset["true_next_label"]))
    return {
        "examples": int(len(choice)),
        "query_label_errors": query_errors,
        "query_label_error_rate": query_errors / len(choice),
        "next_label_errors": next_errors,
        "next_label_error_rate": next_errors / len(choice),
        # Position preference: which of the two context slots was copied.
        "chose_context_position_0": float(np.mean(choice == 0)),
        # Symbol-identity spread: a different question from position preference.
        "distinct_next_symbol_classes": int(len(present)),
        "largest_next_symbol_class_share": float(present.max()),
        "next_symbol_class_entropy_nats": float(-np.sum(present * np.log(present))),
        "generated_label_counts": np.bincount(dataset["labels"][:, 2].astype(np.int64),
                                              minlength=5).tolist(),
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


def train_generation(opts, folder, dataset, features, dev, checkpoints=None):
    """One pass over the dataset, saving checkpoints and the metric log.

    `checkpoints` is the list of sequence counts to save at; the default is the
    agreed 55-point mechanistic schedule. Tuning candidates pass the two
    endpoints instead, because only their final model is compared.
    """
    model = extended.build_model(opts, SYMBOL_HEAD_INIT_SEED)
    optimizer = main_utils.get_optimizer_from_opts(opts)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    _, train_model_seed = common.training_seeds(opts)

    checkpoint_folder = folder / "checkpoints"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    schedule = set(checkpoints if checkpoints is not None
                   else checkpoint_schedule(opts.train_iters, opts.train_bs))

    class_idxs = jnp.asarray(dataset["class_idxs"], jnp.int32)
    exemplar_idxs = jnp.asarray(dataset["exemplar_idxs"], jnp.int32)
    labels = jnp.asarray(dataset["labels"], jnp.int32)
    choice = jnp.asarray(dataset["symbol_choice"], jnp.int32)
    next_label = jnp.asarray(dataset["next_label"], jnp.int32)

    log = {name: [] for name in ("train_total", "train_query", "train_symbol",
                                 "train_next", "train_iter")}
    dev_log, eval_iters = [], []

    def save(sequences):
        eqx.tree_serialise_leaves(checkpoint_folder / "{:011d}.eqx".format(sequences), model)

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
                 label_temperature, symbol_temperature=1.0, context_temperature=None,
                 context_feedback=None):
    """One record per run: the settings, the seeds and where the parent came from.

    `context_feedback` carries the small per-class records that produced this
    run's opening questions, so the sampling weights live beside the run they
    built rather than in a separate file.
    """
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
        next_symbol_temperature=symbol_temperature,
        next_symbol_rule=("sampled from the two-way head at temperature "
                          "{:g}".format(symbol_temperature)),
        context_selection_temperature=context_temperature,
        context_rule=("original task distribution over the 50 training classes"
                      if context_temperature is None else
                      "classes drawn from the parent's own generated next-symbol "
                      "frequencies raised to the power 1/{:g}".format(context_temperature)),
        context_feedback=context_feedback,
        label_strategy=label_strategy,
        label_temperature=(label_temperature if label_strategy == "sample" else None),
        label_rule=("argmax over the five labels" if label_strategy == "argmax" else
                    "sampled from softmax(logits / {}) over all five labels".format(
                        label_temperature)),
        dev_evaluator=str(common.DEV_EVALUATOR_FILE.relative_to(common.ROOT)))
    (folder / "config.json").write_bytes(
        json.dumps(config, indent=2, default=str).encode("utf-8"))


def require_matching_rate(folder, learning_rate):
    """Refuse to reuse a finished run that was trained at a different rate."""
    config = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8"))
    if float(config["lr"]) != learning_rate:
        raise SystemExit("{} was trained at lr {:g}, but this chain runs at {:g}. Move it "
                         "aside before continuing.".format(folder, float(config["lr"]),
                                                           learning_rate))


def temperature_value(text):
    """Parse a temperature, accepting '1/3' so thirds stay exact rather than rounded."""
    if "/" in text:
        top, bottom = text.split("/")
        return float(top) / float(bottom)
    return float(text)


def temperature_name(value):
    """Folder-safe name for a temperature; a third is spelled out, not rounded."""
    return "one_third" if abs(value - 1 / 3) < 1e-9 else "{:g}".format(value)


def condition_name(label_strategy, label_temperature, symbol_temperature=1.0,
                   context_temperature=None):
    """Folder name for a condition.

    The reference condition is `label_argmax`: original opening questions, argmax
    labels, next symbol at temperature 1. Each family changes one thing from it,
    and the name says which.
    """
    if context_temperature is not None:
        return "context_feedback_temperature_" + temperature_name(context_temperature)
    if symbol_temperature != 1.0:
        return "symbol_sampling_temperature_" + temperature_name(symbol_temperature)
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
    parser.add_argument("--label-temperature", type=temperature_value, default=3.0,
                        help="Temperature used when --label-strategy is 'sample'")
    parser.add_argument("--next-symbol-temperature", type=temperature_value, default=1.0,
                        help="Temperature for picking between the two context symbols")
    parser.add_argument("--context-temperature", type=temperature_value, default=None,
                        help="Turn on context feedback: build each child's questions from "
                             "the parent's own generated symbol frequencies, raised to "
                             "the power 1/T. Accepts '1/3'.")
    args = parser.parse_args()
    common.use_above_normal_priority()

    # The authors' published settings, at the learning rate the extended-task
    # search selected. Nothing but label generation differs between conditions.
    opts = common.baseline_options()
    opts.model_output_classes = opts.fs_relabel
    opts.lr = selected_learning_rate()
    print("learning rate:", "{:g}".format(opts.lr), flush=True)
    features = common.load_features()
    splits = main_utils.get_splits_from_opts(opts, features.shape)
    dev = load_dev_set(opts)

    condition = EXPERIMENTS / condition_name(args.label_strategy, args.label_temperature,
                                             args.next_symbol_temperature,
                                             args.context_temperature)
    condition.mkdir(parents=True, exist_ok=True)
    print("condition:", condition.relative_to(common.ROOT), flush=True)

    # Generation 0 is shared: train it once, then reuse it for every condition.
    if (GENERATION_0 / "log.h5").exists():
        require_matching_rate(GENERATION_0, opts.lr)
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
    train_classes = np.asarray(splits["class"]["train"])

    for generation in range(1, args.generations + 1):
        folder = condition / "generation_{}".format(generation)
        if (folder / "log.h5").exists():
            # Refuse to build on a generation trained at a different rate rather
            # than silently mixing settings within one chain.
            require_matching_rate(folder, opts.lr)
            print("generation", generation, "already trained; loading it", flush=True)
            parent_model = load_final_model(folder, opts)
            parent_folder = folder
            continue
        folder.mkdir(parents=True, exist_ok=True)

        # Context feedback: measure what the parent generates on its own saved
        # questions, sharpen those frequencies, and build the child's questions
        # from them. Otherwise the original task distribution is used.
        weights, feedback = None, None
        if args.context_temperature is not None:
            parent_data = load_dataset(parent_folder / "training_data.h5")
            frequencies = measure_generated_symbol_frequencies(
                opts, parent_model, features, parent_data, train_classes,
                args.next_symbol_temperature)
            weights = context_sampling_weights(frequencies, args.context_temperature)
            feedback = {
                "measured_on": str(parent_folder.relative_to(common.ROOT)),
                "frequency_seed": FREQUENCY_SEED,
                "train_classes": train_classes.tolist(),
                "parent_generated_frequencies": frequencies.tolist(),
                "sampling_weights": weights.tolist(),
            }
            print("generation", generation, "context weights: max {:.5f} min {:.5f}".format(
                weights.max(), weights.min()), flush=True)

        dataset = build_successor_dataset(opts, splits, parent_model, features,
                                          opts.train_iters, args.label_strategy,
                                          args.label_temperature,
                                          args.next_symbol_temperature, weights)
        save_dataset(folder / "training_data.h5", dataset)
        quality = dataset_quality(dataset)
        (folder / "dataset_quality.json").write_bytes(
            json.dumps(quality, indent=2).encode("utf-8"))
        print("generation", generation, "dataset:", json.dumps(quality), flush=True)

        write_config(folder, opts, generation, parent_folder, args.label_strategy,
                     args.label_temperature, args.next_symbol_temperature,
                     args.context_temperature, feedback)
        parent_model = train_generation(opts, folder, dataset, features, dev)
        parent_folder = folder
        print("generation", generation, "trained", flush=True)

    print("done;", condition.relative_to(common.ROOT), "holds generations 1 to",
          args.generations)


def load_dataset(path):
    """Read a saved training dataset back into plain arrays."""
    with h5py.File(path, "r") as handle:
        return {name: handle[name][:] for name in handle}


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
