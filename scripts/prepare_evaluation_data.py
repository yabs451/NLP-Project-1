"""Choose assignment validation/final-test classes and build the dev evaluator.

The authors' class split is 50 train / 1473 val / 100 test (contiguous row
ranges). Their `fsl_test_class` evaluator reads the last 100 rows and is scored
throughout training, so those rows cannot serve as an untouched final test. The
1473-row middle pool is unused by every original evaluator, so we draw both of
our assignment splits from it.

Outputs:
  results/evaluation_data/class_splits.json  exact class IDs, rules and seeds
  results/evaluation_data/eval_dev.h5        1,000 fixed dev sequences ('fsl_dev_class')

The final-test classes are recorded but no sequences are generated for them, so
no model can be scored on them by accident. `--build-final-test` regenerates
that file deterministically when the project is ready to use it.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, EVALUATION_DATA, baseline_options

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import main_utils
import samplers

# Seed for our class draw only. Distinct from the authors' init (5), train (0),
# eval (1) and label-pair (20) seeds so it cannot perturb any of their streams.
ASSIGNMENT_CLASS_SPLIT_SEED = 7
DEV_CLASSES = 100
FINAL_TEST_CLASSES = 100
DEV_EVALUATOR_NAME = "fsl_dev_class"
FINAL_TEST_EVALUATOR_NAME = "fsl_final_test_class"


def select_classes(opts, num_rows):
    """Split the authors' unused middle class pool into dev and final-test halves.

    Rule: permute the pool with one fixed key, take the first 100 rows as dev
    validation and the next 100 as final test, then store both sorted.
    """
    splits = main_utils.get_splits_from_opts(opts, (num_rows, 5))
    pool = np.asarray(splits["class"]["val"])
    permuted = np.asarray(jax.random.permutation(
        jax.random.PRNGKey(ASSIGNMENT_CLASS_SPLIT_SEED), pool))
    dev = np.sort(permuted[:DEV_CLASSES])
    final_test = np.sort(permuted[DEV_CLASSES:DEV_CLASSES + FINAL_TEST_CLASSES])
    train = np.asarray(splits["class"]["train"])
    upstream_test = np.asarray(splits["class"]["test"])
    # The disjointness that the protocol depends on, checked rather than assumed.
    assert len(set(dev.tolist())) == DEV_CLASSES
    assert len(set(final_test.tolist())) == FINAL_TEST_CLASSES
    assert not (set(dev.tolist()) & set(final_test.tolist())), "dev/final-test overlap"
    for name, other in (("train", train), ("upstream_test", upstream_test)):
        assert not (set(dev.tolist()) & set(other.tolist())), "dev overlaps " + name
        assert not (set(final_test.tolist()) & set(other.tolist())), "final test overlaps " + name
    return {"dev": dev, "final_test": final_test, "train": train,
            "upstream_test": upstream_test, "pool": pool, "splits": splits}


def build_evaluator(opts, splits, classes, features, seed):
    """Sample fixed evaluation sequences exactly as main.py builds an evaluator.

    Same two-pair supported-query task, exemplar 0 and training label pairs as
    the authors' evaluators; only the class pool differs.
    """
    class_sampler = partial(samplers.get_constant_burst_seq_idxs,
                            classes=jnp.asarray(classes),
                            class_distr=jnp.ones(len(classes)) / len(classes),
                            num_seqs=opts.eval_iters,
                            context_len=opts.train_context_len,
                            burstiness=1, distractor=1, no_support=0, unique_rest=0)
    exemplar_sampler = partial(samplers.get_exemplar_inds,
                               allowed_inds=splits["exemplar"]["train"],
                               match_query_and_distractors=opts.match_query_and_distractors)
    sampler = samplers.make_data_sampler(
        class_sampler, exemplar_sampler,
        fs_relabel=splits["relabeling"]["train"],
        noise_scale=opts.noise_scale_train,
        assign_query_label_random=0)
    return sampler(jax.random.PRNGKey(seed), features)


def check_evaluator(batch, features, classes, opts, allowed_pairs):
    """Confirm the generated sequences really are the intended task."""
    examples = np.asarray(batch["examples"])
    labels = np.asarray(batch["labels"])
    assert examples.shape == (opts.eval_iters, opts.train_context_len + 1, features.shape[-1])
    assert labels.shape == (opts.eval_iters, opts.train_context_len + 1)
    # Exemplar 0 only, so a support symbol is bit-identical to the query symbol.
    support = np.all(examples[:, :-1] == examples[:, -1:], axis=-1)
    assert np.all(support.sum(axis=1) == 1), "each sequence needs exactly one support"
    assert np.all(labels[:, :-1][support] == labels[:, -1]), "support label must match target"
    assert np.all(labels[:, 0] != labels[:, 1]), "context labels must differ"
    allowed = set(map(tuple, np.asarray(allowed_pairs)))
    assert set(map(tuple, np.sort(labels[:, :2], axis=1))) <= allowed
    # Every symbol used must be one of the selected classes' exemplar-0 vectors.
    pool_rows = {row.tobytes() for row in features[np.asarray(classes), 0]}
    used = {row.tobytes() for row in examples.reshape(-1, features.shape[-1])}
    assert used <= pool_rows, "symbols drawn outside the selected class pool"
    return {"sequences": int(labels.shape[0]), "tokens_per_sequence": int(labels.shape[1]),
            "single_support_per_sequence": True, "support_label_matches_target": True,
            "context_labels_distinct": True, "label_pairs_from_training_split": True,
            "symbols_from_selected_classes_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-final-test", action="store_true",
                        help="Also write the reserved final-test sequences (do not use for model selection)")
    args = parser.parse_args()

    output = EVALUATION_DATA
    output.mkdir(parents=True, exist_ok=True)
    opts = baseline_options()
    with h5py.File(opts.data_file, "r") as handle:
        features = jnp.asarray(handle[opts.data_path_in_file][:])
    selection = select_classes(opts, features.shape[0])
    splits = selection["splits"]
    train_pairs = splits["relabeling"]["train"]

    # Separate evaluation seeds per split, both distinct from the authors' eval seed.
    dev_seed = 1000 + ASSIGNMENT_CLASS_SPLIT_SEED
    final_test_seed = 2000 + ASSIGNMENT_CLASS_SPLIT_SEED

    dev = build_evaluator(opts, splits, selection["dev"], features, dev_seed)
    dev_checks = check_evaluator(dev, np.asarray(features), selection["dev"], opts, train_pairs)
    dev_path = output / "eval_dev.h5"
    with h5py.File(dev_path, "w") as handle:
        for field in ("examples", "labels"):
            handle.create_dataset("/".join([DEV_EVALUATOR_NAME, field]), data=np.asarray(dev[field]))

    final_test_path = output / "eval_final_test.h5"
    if args.build_final_test:
        final = build_evaluator(opts, splits, selection["final_test"], features, final_test_seed)
        check_evaluator(final, np.asarray(features), selection["final_test"], opts, train_pairs)
        with h5py.File(final_test_path, "w") as handle:
            for field in ("examples", "labels"):
                handle.create_dataset("/".join([FINAL_TEST_EVALUATOR_NAME, field]),
                                      data=np.asarray(final[field]))

    record = {
        "purpose": "Assignment validation/final-test class protocol, fixed before any model evaluation.",
        "source_pool": {"name": "authors' unused 'val' class range",
                        "class_split": list(opts.class_split),
                        "rows": [int(selection["pool"][0]), int(selection["pool"][-1])],
                        "size": int(len(selection["pool"]))},
        "selection_rule": ("jax.random.permutation of the 1473-row pool with "
                           "PRNGKey(ASSIGNMENT_CLASS_SPLIT_SEED); first 100 rows are development "
                           "validation, next 100 are final test; both stored sorted"),
        "seeds": {"class_split": ASSIGNMENT_CLASS_SPLIT_SEED,
                  "dev_evaluator_data": dev_seed,
                  "final_test_evaluator_data": final_test_seed,
                  "authors_untouched": {"init_seed": opts.init_seed, "train_seed": opts.train_seed,
                                        "eval_seed": opts.eval_seed,
                                        "fs_relabel_split_seed": opts.fs_relabel_split_seed}},
        "dev_class_ids": selection["dev"].tolist(),
        "final_test_class_ids": selection["final_test"].tolist(),
        "train_class_ids": selection["train"].tolist(),
        "upstream_test_class_ids_not_used_by_us": [int(selection["upstream_test"][0]),
                                                   int(selection["upstream_test"][-1])],
        "disjointness_checked": {
            "dev_vs_final_test": True, "dev_vs_train": True, "final_test_vs_train": True,
            "dev_vs_upstream_test_class_evaluator": True,
            "final_test_vs_upstream_test_class_evaluator": True,
            "note": ("The previous trial scored fsl_test_class on rows 1523-1622 only. "
                     "Neither assignment split touches those rows or the 50 training rows.")},
        "task_settings": {"context_len": int(opts.train_context_len), "burstiness": 1,
                          "distractor": True, "exemplar_split": "train (exemplar 0 only)",
                          "label_pairs": "training split", "sequences": int(opts.eval_iters),
                          "output_classes": int(opts.fs_relabel)},
        "dev_evaluator": {"name": DEV_EVALUATOR_NAME, "file": str(dev_path.relative_to(ROOT)),
                          "checks": dev_checks},
        "final_test_evaluator": {
            "name": FINAL_TEST_EVALUATOR_NAME,
            "file": str(final_test_path.relative_to(ROOT)),
            "built": bool(args.build_final_test),
            "status": ("RESERVED - regenerate deterministically with --build-final-test when the "
                       "project is ready; no model has been scored on it")},
    }
    (output / "class_splits.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("dev_class_ids", "final_test_class_ids", "train_class_ids")}, indent=2))
    print("dev classes (first 10):", record["dev_class_ids"][:10])
    print("final-test classes (first 10):", record["final_test_class_ids"][:10])


if __name__ == "__main__":
    main()
