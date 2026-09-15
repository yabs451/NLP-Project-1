"""Reload checkpoints, check parameter updates and replay saved evaluations."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ["JAX_PLATFORMS"] = "cpu"
sys.dont_write_bytecode = True
from run_baseline import UPSTREAM
sys.path.insert(0, str(UPSTREAM))

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import main as training
import main_utils
import opto


def load_checkpoint(path, opts):
    model = main_utils.get_model_from_opts(argparse.Namespace(**vars(opts)))
    optimizer = main_utils.get_optimizer_from_opts(opts)
    template = {"iter": -1, "model": model,
        "opt_state": optimizer.init(eqx.filter(model, eqx.is_array)),
        "seeds": {name: jax.random.PRNGKey(0) for name in
                  ("eval_model_seed", "train_data_seed", "train_model_seed")}}
    return eqx.tree_deserialise_leaves(path, template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    args = parser.parse_args()
    folder = args.run_folder.resolve()
    opts = argparse.Namespace(**json.loads((folder / "config.json").read_text()))
    initial = load_checkpoint(folder / "checkpoints" / "00000000000.eqx", opts)
    final_path = folder / "checkpoints" / f"{opts.train_iters:011d}.eqx"
    final = load_checkpoint(final_path, opts)
    assert final["iter"] == opts.train_iters
    before = jax.tree_util.tree_leaves(eqx.filter(initial["model"], eqx.is_array))
    after = jax.tree_util.tree_leaves(eqx.filter(final["model"], eqx.is_array))
    assert all(np.isfinite(np.asarray(x)).all() for x in after)
    delta = float(np.sqrt(sum(np.sum((np.asarray(a, dtype=np.float64) - np.asarray(b)) ** 2)
                              for a, b in zip(after, before))))
    assert delta > 0, "Training did not change parameters"
    assert int(final["opt_state"][0].count) == opts.train_iters // opts.train_bs
    with h5py.File(folder / "eval_data.h5", "r") as handle:
        eval_data = {name: {field: jnp.asarray(handle[name][field][:]) for field in ("examples", "labels")}
                     for name in opts.pe_names}
    forward = opto.make_fn_from_opts(opts)
    assert forward is opto.default_model_fwd_fn, "Baseline unexpectedly uses an intervention"
    start = perf_counter()
    replay = training.evaluate(final["model"], forward, jax.random.PRNGKey(321), eval_data, opts.eval_bs)
    jax.block_until_ready(replay)
    replay_seconds = perf_counter() - start
    metrics = {}
    with h5py.File(folder / "log.h5", "r") as handle:
        assert len(handle["train_loss"]) == opts.train_iters // opts.train_bs
        for field in ("train_loss", "train_grad_norm", "train_grad_batch_stddev"):
            assert np.isfinite(handle[field][:]).all()
        for name in opts.pe_names:
            for field, value in replay[name].items():
                np.testing.assert_allclose(value, handle[name][field][-1], rtol=1e-5, atol=1e-6)
            metrics[name] = {field: [float(row.mean()) for row in handle[name][field][:]]
                             for field in ("loss", "acc", "in_context_acc", "prob", "use_context_prob")}
        train_loss = handle["train_loss"][:]
        summary = {"eval_sequence_counts": handle["eval_iter"][:].tolist(), "metrics": metrics,
            "train_loss_first": float(train_loss[0]), "train_loss_last": float(train_loss[-1]),
            "train_loss_first_10_mean": float(train_loss[:10].mean()),
            "train_loss_last_10_mean": float(train_loss[-10:].mean()),
            "train_grad_norm_min": float(handle["train_grad_norm"][:].min()),
            "train_grad_norm_max": float(handle["train_grad_norm"][:].max())}
    # The target is passed for scoring, but must never enter the input tokens.
    sample = eval_data[opts.pe_names[0]]
    x, y, key = sample["examples"][0], sample["labels"][0], jax.random.PRNGKey(9)
    normal = forward(final["model"], x, y, key)
    changed = forward(final["model"], x, y.at[-1].set((y[-1] + 1) % opts.model_output_classes), key)
    np.testing.assert_array_equal(normal["out"], changed["out"])
    assert normal["tok_embedding"].shape == (5, 64)
    # Check that the loaded optimizer can perform another finite update in memory.
    optimizer = main_utils.get_optimizer_from_opts(opts)
    step_metrics, resumed, _ = training.train_step(final["model"], forward, optimizer,
        final["opt_state"], opts.train_microbs, opts.weight_decay,
        sample["examples"][:32], sample["labels"][:32], jax.random.PRNGKey(10))
    assert all(np.isfinite(np.asarray(value)).all() for value in step_metrics.values())
    resumed_leaves = jax.tree_util.tree_leaves(eqx.filter(resumed, eqx.is_array))
    assert all(np.isfinite(np.asarray(value)).all() for value in resumed_leaves)
    assert any(not np.array_equal(a, b) for a, b in zip(after, resumed_leaves))
    summary.update(checks_passed=True, checkpoint=str(final_path), checkpoint_bytes=final_path.stat().st_size,
        parameter_count=sum(x.size for x in after), parameter_change_l2=delta,
        optimizer_updates=int(final["opt_state"][0].count), checkpoint_eval_replay_seconds=replay_seconds,
        checkpoint_eval_matches_log=True, query_target_excluded_from_input=True,
        restored_optimizer_update_passed=True, verification_update_saved=False)
    (folder / "verification.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
