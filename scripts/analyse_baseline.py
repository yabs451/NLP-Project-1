"""Curves, induction-circuit progress measures, attention maps and head ablations.

All mechanistic quantities reuse the authors' own forward pass,
`visualize_runs.make_forward_fn`, which returns per-sequence attention scores
shaped [batch, layer, head, query_position, key_position] together with loss,
accuracy and context-restricted accuracy.

The two progress measures follow the authors' plotting code:

* Previous-token score (layer 0), from `update_prev_token_over_time` /
  `plot_prev_token_over_time` in upstream/icl-dynamics/visualize_runs.py:338-364.
  Raw score is attention from token i to token i-1. The authors subtract a
  chance baseline: with attention a on the previous token, the remaining 1-a is
  spread over the i other allowed positions, so the corrected score is
  a - (1-a)/(i+1). We report the average over all i and, separately, the
  average over label tokens 1 and 3 (the authors' "Average for 1,3" row), which
  is the part the induction circuit needs.

* Induction score (layer 1), from `plot_attention_over_time`
  (visualize_runs.py:513-526). At the query token, the correct label token sits
  at position 2*correct_ind+1 and the incorrect one at 2*(1-correct_ind)+1. We
  report attention to the correct label token and the authors' delta,
  correct - incorrect. `opto.py:396` uses the same convention.

Ablation means the authors' `--opto_ablate_heads layer:head`
(opto.py:368-372): that head's value vectors are set to zero, so it writes
nothing into the residual stream while every other head is untouched.

Run:
  .venv/Scripts/python.exe scripts/analyse_baseline.py results/<run>
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.dont_write_bytecode = True
from run_baseline import ROOT, UPSTREAM
sys.path.insert(0, str(UPSTREAM))

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import main_utils
import visualize_runs

# 5 tokens: symbol, label, symbol, label, query symbol.
QUERY_TOKEN = 4
NUM_CHECKPOINTS_TO_ANALYSE = 15


def load_dev_data(folder, opts):
    """Load the fixed development evaluator this run was scored on."""
    path = ROOT / "results" / "assignment" / "eval_dev.h5"
    with h5py.File(path, "r") as handle:
        name = list(handle.keys())[0]
        data = {field: jnp.asarray(handle[name][field][:]) for field in ("examples", "labels")}
    # Which context pair holds the support symbol. Exemplar 0 only, so the
    # support symbol vector is bit-identical to the query symbol vector.
    examples = np.asarray(data["examples"])
    matches_first = np.all(examples[:, 0] == examples[:, 2], axis=-1)
    correct_ind = np.where(matches_first, 0, 1)
    assert np.all(np.all(examples[:, 1] == examples[:, 2], axis=-1) == (correct_ind == 1))
    return name, data, correct_ind


def checkpoint_iters(folder):
    """A modest, roughly log-spaced selection spanning the whole run."""
    available = sorted(int(p.stem) for p in (folder / "checkpoints").glob("*.eqx"))
    targets = np.unique(np.concatenate([[0], np.geomspace(
        max(available[1], 1), available[-1], NUM_CHECKPOINTS_TO_ANALYSE - 1)]))
    chosen = sorted({available[int(np.abs(np.asarray(available) - t).argmin())] for t in targets})
    return available, chosen


def load_model(folder, iteration, opts):
    model = main_utils.get_model_from_opts(argparse.Namespace(**vars(opts)))
    optimizer = main_utils.get_optimizer_from_opts(opts)
    template = {"iter": -1, "model": model,
                "opt_state": optimizer.init(eqx.filter(model, eqx.is_array)),
                "seeds": {name: jax.random.PRNGKey(0) for name in
                          ("eval_model_seed", "train_data_seed", "train_model_seed")}}
    ckpt = eqx.tree_deserialise_leaves(folder / "checkpoints" / f"{iteration:011d}.eqx", template)
    assert ckpt["iter"] == iteration
    return ckpt


def previous_token_scores(attention):
    """Layer-0 attention from each token to its predecessor, chance-corrected.

    attention: [batch, layer, head, query_pos, key_pos].
    Returns raw and corrected arrays shaped [pair_index, head, batch].
    """
    batch, _, heads, seq, _ = attention.shape
    rows = np.arange(1, seq)
    # Same indexing as upstream. NumPy puts the paired advanced-index axis first,
    # giving [pair, batch, head]; we reorder to [pair, head, batch].
    raw = attention[:, 0, :, rows, rows - 1]
    assert raw.shape == (seq - 1, batch, heads), raw.shape
    raw = raw.transpose(0, 2, 1)
    # Chance baseline: the 1-a that does not go to the previous token is spread
    # over the pair_index+1 other causally allowed positions.
    corrected = raw - (1 - raw) / (1 + np.arange(seq - 1))[:, None, None]
    return raw, corrected


def induction_scores(attention, correct_ind):
    """Layer-1 attention from the query to the correct vs incorrect label token."""
    query_row = np.asarray(attention[:, 1, :, QUERY_TOKEN, :])  # [batch, head, key_pos]
    batch = np.arange(query_row.shape[0])
    correct = query_row[batch, :, 2 * correct_ind + 1]
    incorrect = query_row[batch, :, 2 * (1 - correct_ind) + 1]
    return correct, incorrect


def measure_checkpoint(model, forward, data, correct_ind, key):
    results = forward(model, data["examples"], data["labels"], key=key)
    attention = np.asarray(results["activations"])
    raw_prev, corrected_prev = previous_token_scores(attention)
    correct, incorrect = induction_scores(attention, correct_ind)
    return {
        "attention": attention,
        "prev_token_all_positions": corrected_prev.mean(axis=(0, 2)),
        "prev_token_label_positions": corrected_prev[0::2].mean(axis=(0, 2)),
        "prev_token_raw_label_positions": raw_prev[0::2].mean(axis=(0, 2)),
        "induction_correct": correct.mean(axis=0),
        "induction_delta": (correct - incorrect).mean(axis=0),
        "acc": float(np.mean(results["acc"])),
        "in_context_acc": float(np.mean(results["in_context_acc"])),
        "loss": float(np.mean(results["loss"])),
    }


def plot_curves(log, evaluators, output):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    x = log["eval_iter"]
    for name in evaluators:
        axs[0].plot(x, log[name]["loss"], label=name)
        axs[1].plot(x, log[name]["acc"], label=name)
        axs[2].plot(x, log[name]["in_context_acc"], label=name)
    axs[0].plot(log["train_x"], log["train_loss_smooth"], "k--", alpha=0.6, label="train (smoothed)")
    axs[0].set_ylabel("query cross-entropy (nats)")
    axs[1].axhline(0.2, color="grey", ls=":", label="chance (1 of 5)")
    axs[1].set_ylabel("accuracy over 5 labels")
    axs[2].axhline(0.5, color="grey", ls=":", label="chance (1 of 2)")
    axs[2].set_ylabel("accuracy restricted to in-context labels")
    for ax, title in zip(axs, ("Loss", "Accuracy", "Context-restricted accuracy")):
        ax.set_title(title)
        ax.set_xlabel("training sequences")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)


def plot_head_measures(iters, measures, output):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    heads = len(measures[0]["induction_delta"])
    for head in range(heads):
        axs[0].plot(iters, [m["prev_token_label_positions"][head] for m in measures], label=f"L0H{head}")
        axs[1].plot(iters, [m["induction_delta"][head] for m in measures], label=f"L1H{head}")
    axs[2].plot(iters, [m["acc"] for m in measures], label="dev accuracy")
    axs[2].plot(iters, [m["in_context_acc"] for m in measures], label="dev context-restricted")
    axs[2].axhline(0.2, color="grey", ls=":")
    axs[2].axhline(0.5, color="grey", ls="--")
    axs[0].set_title("Layer 0: previous-token score\n(label tokens 1 and 3, chance-corrected)")
    axs[1].set_title("Layer 1: induction score\n(query attention, correct - incorrect label)")
    axs[2].set_title("Development-set behaviour")
    for ax in axs:
        ax.set_xlabel("training sequences")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "head_measures.png", dpi=150)
    plt.close(fig)


def plot_attention_example(attention, labels, correct_ind, output):
    """One development sequence, every head in both layers."""
    ticks = ["sym A", "lab A", "sym B", "lab B", "query"]
    heads = attention.shape[2]
    fig, axs = plt.subplots(2, heads, figsize=(2.1 * heads, 5.6))
    for layer in range(2):
        for head in range(heads):
            ax = axs[layer, head]
            image = ax.imshow(attention[0, layer, head], vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"L{layer}H{head}", fontsize=8)
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            # Only the bottom row and first column carry tick labels, so the
            # per-panel titles stay readable.
            ax.set_xticklabels(ticks if layer == 1 else [""] * 5, rotation=90, fontsize=6)
            ax.set_yticklabels(ticks if head == 0 else [""] * 5, fontsize=6)
    fig.colorbar(image, ax=axs, shrink=0.6, label="attention weight")
    support = "A" if correct_ind[0] == 0 else "B"
    fig.suptitle("Final-checkpoint attention, one development sequence "
                 f"(support = symbol {support}, target label {int(labels[0, -1])}); "
                 "rows attend to columns", fontsize=9)
    fig.savefig(output / "attention_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def ablation(opts, model, data, correct_ind, heads, key):
    """Score the model with the given heads' value vectors zeroed."""
    ablated_opts = argparse.Namespace(**copy.deepcopy(vars(opts)))
    ablated_opts.opto_ablate_heads = heads
    forward = visualize_runs.make_forward_fn(ablated_opts)
    results = forward(model, data["examples"], data["labels"], key=key)
    return {"heads": heads, "acc": float(np.mean(results["acc"])),
            "in_context_acc": float(np.mean(results["in_context_acc"])),
            "loss": float(np.mean(results["loss"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    args = parser.parse_args()
    folder = args.run_folder.resolve()
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    opts = argparse.Namespace(**json.loads((folder / "config.json").read_text()))

    # --- learning curves from the run's own log ---
    log = {}
    with h5py.File(folder / "log.h5", "r") as handle:
        evaluators = sorted(k for k in handle.keys() if k.startswith("fsl"))
        log["eval_iter"] = handle["eval_iter"][:]
        for name in evaluators:
            log[name] = {field: handle[name][field][:].mean(axis=1)
                         for field in ("loss", "acc", "in_context_acc")}
        train_loss = handle["train_loss"][:]
    window = max(1, len(train_loss) // 200)
    trimmed = train_loss[:len(train_loss) // window * window].reshape(-1, window)
    log["train_loss_smooth"] = trimmed.mean(axis=1)
    log["train_x"] = (np.arange(len(trimmed)) + 1) * window * opts.train_bs
    plot_curves(log, evaluators, output)

    # --- mechanistic measures across checkpoints ---
    name, data, correct_ind = load_dev_data(folder, opts)
    forward = visualize_runs.make_forward_fn(argparse.Namespace(**vars(opts)))
    available, chosen = checkpoint_iters(folder)
    key = jax.random.PRNGKey(0)
    measures = []
    for iteration in chosen:
        ckpt = load_model(folder, iteration, opts)
        measures.append(measure_checkpoint(ckpt["model"], forward, data, correct_ind, key))
        print(f"{iteration:>9} dev acc {measures[-1]['acc']:.3f} "
              f"in-context {measures[-1]['in_context_acc']:.3f} "
              f"max induction delta {measures[-1]['induction_delta'].max():+.3f}", flush=True)
    plot_head_measures(chosen, measures, output)

    final = measures[-1]
    plot_attention_example(final["attention"], np.asarray(data["labels"]), correct_ind, output)

    # --- pick candidate heads from observed behaviour, not fixed indices ---
    induction_head = int(np.argmax(final["induction_delta"]))
    comparison_head = int(np.argmin(np.abs(final["induction_delta"])))
    prev_token_head = int(np.argmax(final["prev_token_label_positions"]))
    prev_comparison_head = int(np.argmin(final["prev_token_label_positions"]))

    final_model = load_model(folder, available[-1], opts)["model"]
    ablations = {
        "intact": ablation(opts, final_model, data, correct_ind, None, key),
        "candidate_induction_head_ablated": ablation(
            opts, final_model, data, correct_ind, [f"1:{induction_head}"], key),
        "comparison_layer1_head_ablated": ablation(
            opts, final_model, data, correct_ind, [f"1:{comparison_head}"], key),
        "candidate_previous_token_head_ablated": ablation(
            opts, final_model, data, correct_ind, [f"0:{prev_token_head}"], key),
        "comparison_layer0_head_ablated": ablation(
            opts, final_model, data, correct_ind, [f"0:{prev_comparison_head}"], key),
    }

    summary = {
        "run": str(folder.relative_to(ROOT)),
        "dev_evaluator": name,
        "checkpoints_available": len(available),
        "final_checkpoint_iter": int(available[-1]),
        "final_checkpoint_matches_configured_iters": int(available[-1]) == int(opts.train_iters),
        "optimizer_updates": int(opts.train_iters // opts.train_bs),
        "checkpoints_analysed": [int(i) for i in chosen],
        "candidate_heads": {
            "induction_layer1_head": induction_head,
            "comparison_layer1_head": comparison_head,
            "previous_token_layer0_head": prev_token_head,
            "comparison_layer0_head": prev_comparison_head,
            "selected_by": "largest/smallest observed measure at the final checkpoint, not hardcoded"},
        "final_measures": {
            "dev_acc": final["acc"], "dev_in_context_acc": final["in_context_acc"],
            "dev_loss": final["loss"],
            "induction_delta_per_head_layer1": final["induction_delta"].tolist(),
            "induction_correct_attention_per_head_layer1": final["induction_correct"].tolist(),
            "prev_token_label_positions_per_head_layer0": final["prev_token_label_positions"].tolist(),
            "prev_token_all_positions_per_head_layer0": final["prev_token_all_positions"].tolist()},
        "trajectory": [
            {"iter": int(i), "dev_acc": m["acc"], "dev_in_context_acc": m["in_context_acc"],
             "dev_loss": m["loss"],
             "max_induction_delta_layer1": float(m["induction_delta"].max()),
             "max_prev_token_label_layer0": float(m["prev_token_label_positions"].max())}
            for i, m in zip(chosen, measures)],
        "ablations": ablations,
        "ablation_meaning": ("opto.py sets the listed heads' value vectors to zero, so they "
                             "contribute nothing to the residual stream; all other heads and "
                             "all weights are unchanged."),
        "chance_levels": {"acc": 0.2, "in_context_acc": 0.5},
    }
    (output / "analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "trajectory"}, indent=2))


if __name__ == "__main__":
    main()
