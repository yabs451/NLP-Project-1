"""Check that a finished training run is sound, for any generation.

Confirms the things a result depends on: the final checkpoint loads, the
optimizer really performed the expected number of updates, parameters are
finite and moved away from initialisation, the training log is complete, the
query's target never reaches the model input, and re-scoring the final
checkpoint reproduces the metrics the run itself logged.

Usage (from the project root):
  .venv/Scripts/python.exe scripts/verify_run.py results/base_task/<run>
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import main as upstream_main
import opto


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_folder", type=Path)
    args = parser.parse_args()
    folder = args.run_folder.resolve()
    opts = common.load_run_options(folder)
    # A run's config records the evaluator file path as it was at the time; use
    # the current location so verification works after any reorganisation.
    opts.load_eval_data = [str(common.DEV_EVALUATOR_FILE)]

    checkpoints = common.available_checkpoints(folder)
    initial = common.load_checkpoint(folder, checkpoints[0], opts)
    final = common.load_checkpoint(folder, checkpoints[-1], opts)

    before = jax.tree_util.tree_leaves(eqx.filter(initial["model"], eqx.is_array))
    after = jax.tree_util.tree_leaves(eqx.filter(final["model"], eqx.is_array))
    assert all(np.isfinite(np.asarray(x)).all() for x in after), "non-finite parameters"
    parameter_change = float(np.sqrt(sum(
        np.sum((np.asarray(a, dtype=np.float64) - np.asarray(b)) ** 2)
        for a, b in zip(after, before))))
    assert parameter_change > 0, "training did not change the parameters"

    expected_updates = opts.train_iters // opts.train_bs
    actual_updates = int(final["opt_state"][0].count)
    assert actual_updates == expected_updates, "optimizer update count does not match the schedule"

    # Re-score the final checkpoint on freshly rebuilt evaluators. These are
    # rebuilt from the run's own seeds, so they must be the same questions.
    features = common.load_features()
    evaluators = common.build_evaluators(opts, features)
    forward = opto.make_fn_from_opts(opts)
    assert forward is opto.default_model_fwd_fn, "run unexpectedly used an intervention"
    replay = upstream_main.evaluate(final["model"], forward, jax.random.PRNGKey(0),
                                    evaluators, opts.eval_bs)

    metrics, matches_log = {}, {}
    with h5py.File(folder / "log.h5", "r") as handle:
        assert len(handle["train_loss"]) == expected_updates, "training log is incomplete"
        for field in ("train_loss", "train_grad_norm", "train_grad_batch_stddev"):
            assert np.isfinite(handle[field][:]).all(), field + " contains non-finite values"
        for name in replay:
            matches_log[name] = bool(np.allclose(
                replay[name]["acc"], handle[name]["acc"][-1], rtol=1e-5, atol=1e-6))
            metrics[name] = {"acc": float(np.mean(handle[name]["acc"][-1])),
                             "in_context_acc": float(np.mean(handle[name]["in_context_acc"][-1])),
                             "loss": float(np.mean(handle[name]["loss"][-1]))}
        train_loss = handle["train_loss"][:]
    assert all(matches_log.values()), "re-scoring disagrees with the logged metrics"

    # The target must be used for scoring only, never as model input.
    sample = evaluators[list(evaluators)[0]]
    x, y, key = sample["examples"][0], sample["labels"][0], jax.random.PRNGKey(9)
    normal = forward(final["model"], x, y, key)
    altered = forward(final["model"], x,
                      y.at[-1].set((y[-1] + 1) % opts.model_output_classes), key)
    assert np.array_equal(normal["out"], altered["out"]), "query target reached the model input"

    summary = {
        "run": str(folder.relative_to(common.ROOT)),
        "generation": json.loads((folder / "generation_metadata.json").read_text())["generation"]
                      if (folder / "generation_metadata.json").exists() else 0,
        "checkpoints": len(checkpoints),
        "final_checkpoint_sequences": checkpoints[-1],
        "configured_train_iters": int(opts.train_iters),
        "final_checkpoint_matches_configuration": checkpoints[-1] == int(opts.train_iters),
        "optimizer_updates": actual_updates,
        "expected_optimizer_updates": expected_updates,
        "parameter_count": int(sum(x.size for x in after)),
        "parameter_change_l2_from_initialisation": parameter_change,
        "train_loss_first_10_mean": float(train_loss[:10].mean()),
        "train_loss_last_10_mean": float(train_loss[-10:].mean()),
        "final_metrics_from_log": metrics,
        "rescoring_matches_log": matches_log,
        "query_target_excluded_from_input": True,
        "evaluators_scored": sorted(replay),
        "final_test_evaluator_absent": not any("final_test" in n for n in replay),
        "all_checks_passed": True,
    }
    (folder / "verification.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
