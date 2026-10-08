"""Evaluate the frozen report models once on the reserved final-test evaluator.

This script only loads existing final checkpoints and writes one JSON result
file. It does not train, generate data, compute attention measures, or perform
ablations.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

sys.path.insert(0, str(Path(__file__).resolve().parent / "extended_task"))
import run_extended as extended_pipeline

import h5py
import jax
import jax.numpy as jnp
import numpy as np


FINAL_TEST_FILE = common.EVALUATION_DATA / "eval_final_test.h5"
FINAL_TEST_EVALUATOR_NAME = "fsl_final_test_class"
OUTPUT_FILE = common.ROOT / "results" / "final_test" / "final_test_results.json"

BASE_RECURSIVE = common.ROOT / "results" / "base_task" / "recursive"
EXTENDED_RECURSIVE = common.ROOT / "results" / "extended_task" / "recursive"
EXTENDED_GENERATION_0 = EXTENDED_RECURSIVE / "generation_0"
EXTENDED_EXPERIMENTS = EXTENDED_RECURSIVE / "experiments"

FROZEN_EXTENDED_CONDITIONS = {
    "label_argmax": 4,
    "label_sampling_temperature_3": 4,
    "label_sampling_temperature_5": 4,
    "context_feedback_temperature_one_third": 6,
    "symbol_sampling_temperature_0.2": 6,
}


def final_checkpoint(folder):
    """Return the last saved checkpoint iteration, refusing incomplete paths."""
    folder = Path(folder)
    available = common.available_checkpoints(folder)
    if not available:
        raise SystemExit("No checkpoints found in {}".format(folder.relative_to(common.ROOT)))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    iteration = available[-1]
    if iteration != int(config["train_iters"]):
        raise SystemExit("{} ends at checkpoint {}, expected {}".format(
            folder.relative_to(common.ROOT), iteration, config["train_iters"]))
    return iteration


def frozen_run_inventory():
    """The fixed model set, with shared extended generation 0 listed once."""
    runs = []
    for generation in range(5):
        runs.append({"task": "base", "condition": "base_recursive",
                     "generation": generation,
                     "folder": BASE_RECURSIVE / "generation_{}".format(generation)})
    runs.append({"task": "extended", "condition": "shared_generation_0",
                 "generation": 0, "folder": EXTENDED_GENERATION_0,
                 "shared_by": list(FROZEN_EXTENDED_CONDITIONS)})
    for condition, final_generation in FROZEN_EXTENDED_CONDITIONS.items():
        for generation in range(1, final_generation + 1):
            runs.append({"task": "extended", "condition": condition,
                         "generation": generation,
                         "folder": (EXTENDED_EXPERIMENTS / condition /
                                    "generation_{}".format(generation))})
    return runs


def preflight_runs(runs):
    """Resolve every frozen final checkpoint before any final-test data is read."""
    resolved = []
    for run in runs:
        folder = run["folder"]
        if not folder.is_dir():
            raise SystemExit("Missing frozen run: {}".format(folder.relative_to(common.ROOT)))
        resolved.append({**run, "checkpoint_iteration": final_checkpoint(folder)})
    return resolved


def load_extended_evaluator():
    """Extend the reserved base questions exactly as development evaluation does."""
    with h5py.File(FINAL_TEST_FILE, "r") as handle:
        if list(handle.keys()) != [FINAL_TEST_EVALUATOR_NAME]:
            raise SystemExit("Unexpected evaluator names in {}: {}".format(
                FINAL_TEST_FILE.relative_to(common.ROOT), list(handle.keys())))
        symbols = jnp.asarray(handle[FINAL_TEST_EVALUATOR_NAME]["examples"][:])
        labels = np.asarray(handle[FINAL_TEST_EVALUATOR_NAME]["labels"][:])
    choice = np.asarray(jax.random.bernoulli(
        jax.random.PRNGKey(extended_pipeline.DEV_CHOICE_SEED), 0.5, (len(labels),)),
        np.int32)
    next_label = extended_pipeline.true_continuation_targets(labels, choice)
    return {"symbols": symbols, "labels": jnp.asarray(labels, jnp.int32),
            "symbol_choice": jnp.asarray(choice), "next_label": jnp.asarray(next_label),
            "true_next_label": jnp.asarray(next_label)}


def result_record(run, metrics):
    folder = run["folder"]
    iteration = run["checkpoint_iteration"]
    record = {
        "task": run["task"],
        "condition": run["condition"],
        "generation": run["generation"],
        "run": str(folder.relative_to(common.ROOT)),
        "checkpoint_iteration": iteration,
        "checkpoint": str(common.checkpoint_path(folder, iteration).relative_to(common.ROOT)),
        "metrics": metrics,
    }
    if "shared_by" in run:
        record["shared_by_conditions"] = run["shared_by"]
    return record


def evaluate_base(run):
    folder = run["folder"]
    scored = common.score_checkpoint(
        folder, run["checkpoint_iteration"], FINAL_TEST_FILE,
        opts=common.load_run_options(folder))
    if list(scored) != [FINAL_TEST_EVALUATOR_NAME]:
        raise SystemExit("Unexpected base evaluator names: {}".format(list(scored)))
    metrics = scored[FINAL_TEST_EVALUATOR_NAME]
    return result_record(run, {
        "query_accuracy": metrics["acc"],
        "query_loss": metrics["loss"],
    })


def evaluate_extended(run, opts, evaluator):
    model = extended_pipeline.load_model(
        run["folder"], run["checkpoint_iteration"], opts)
    metrics = extended_pipeline.score_dev(model, evaluator, jax.random.PRNGKey(0))
    return result_record(run, metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    if not FINAL_TEST_FILE.exists():
        raise SystemExit("Missing {}. Build it once with "
                         "scripts/prepare_evaluation_data.py --build-final-test, then run "
                         "this frozen evaluation command."
                         .format(FINAL_TEST_FILE.relative_to(common.ROOT)))

    runs = preflight_runs(frozen_run_inventory())
    extended_opts = common.baseline_options()
    extended_opts.model_output_classes = extended_opts.fs_relabel
    extended_evaluator = load_extended_evaluator()

    models = []
    for run in runs:
        print("scoring", run["folder"].relative_to(common.ROOT), flush=True)
        if run["task"] == "base":
            models.append(evaluate_base(run))
        else:
            models.append(evaluate_extended(run, extended_opts, extended_evaluator))

    output = {
        "result_kind": "reserved_final_test",
        "reserved_final_test": True,
        "evaluation_file": str(FINAL_TEST_FILE.relative_to(common.ROOT)),
        "evaluator_name": FINAL_TEST_EVALUATOR_NAME,
        "checkpoint_rule": "final saved checkpoint, required to equal configured train_iters",
        "scoring": {
            "base": "existing query accuracy and cross-entropy loss definitions",
            "extended": ("existing teacher-forced query-label, symbol-choice and "
                         "following-label definitions; symbol choices use the fixed "
                         "development-evaluation seed"),
        },
        "frozen_extended_conditions": FROZEN_EXTENDED_CONDITIONS,
        "models": models,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("written:", OUTPUT_FILE.relative_to(common.ROOT))


if __name__ == "__main__":
    main()
