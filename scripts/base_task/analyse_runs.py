"""Curves, induction-circuit progress measures, attention maps and head ablations.

All mechanistic quantities reuse the authors' own forward pass,
`visualize_runs.make_forward_fn`, which returns per-sequence attention scores
shaped [batch, layer, head, query_position, key_position] together with loss,
accuracy and context-restricted accuracy.

The two progress measures follow the authors' plotting code:

* Previous-token score (layer 0), from `update_prev_token_over_time` /
  `plot_prev_token_over_time` in upstream/icl-dynamics/visualize_runs.py:338-364.
  Raw score is the attention `a` a token gives to its predecessor. The authors
  subtract a chance baseline, indexed by `r`, the position in the shortened
  array built with `inds = arange(1, seq)`: r=0 is token position 1, r=1 is
  position 2, and so on. The corrected score is `a - (1 - a) / (1 + r)`.
  The denominator `1 + r` counts the causally visible positions OTHER than the
  previous token. For r=1 (token 2) the visible positions are {0,1,2}, the
  previous token is 1, so two others remain and 1+r = 2. The measure is exactly
  zero when attention is uniform over everything the token can see.
  We report the average over all r and, separately, over label tokens 1 and 3
  (the authors' "Average for 1,3" row), which is what the induction circuit
  needs.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
from common import ROOT, DEV_EVALUATOR_FILE

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import visualize_runs

# 5 tokens: symbol, label, symbol, label, query symbol.
QUERY_TOKEN = 4
NUM_CHECKPOINTS_TO_ANALYSE = 15


def load_dev_data(folder, opts):
    """Load the fixed development evaluator this run was scored on."""
    with h5py.File(DEV_EVALUATOR_FILE, "r") as handle:
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
    available = common.available_checkpoints(folder)
    targets = np.unique(np.concatenate([[0], np.geomspace(
        max(available[1], 1), available[-1], NUM_CHECKPOINTS_TO_ANALYSE - 1)]))
    chosen = sorted({available[int(np.abs(np.asarray(available) - t).argmin())] for t in targets})
    return available, chosen


def load_model(folder, iteration, opts):
    """Load one checkpoint. Thin wrapper so the analysis reads naturally."""
    return common.load_checkpoint(folder, iteration, opts)


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
    # Chance baseline: the attention not given to the previous token is spread
    # over the (1 + r) causally visible positions other than the previous token,
    # where r is the index along axis 0. Uniform attention scores exactly zero.
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


def read_generation_summaries(recursive_folder):
    """Collect each generation's analysis.json, in generation order.

    Every generation must already have been analysed individually, so the
    measures being compared were produced by identical code and identical
    checkpoint-selection rules.
    """
    summaries = []
    for folder in sorted(Path(recursive_folder).glob("generation_*"),
                         key=lambda f: int(f.name.split("_")[1])):
        analysis = folder / "analysis" / "analysis.json"
        if not analysis.exists():
            raise SystemExit("{} has not been analysed yet".format(folder))
        summary = json.loads(analysis.read_text(encoding="utf-8"))
        summary["generation"] = int(folder.name.split("_")[1])
        # Successors record how good the parent's generated answers were; that
        # is the quantity most likely to explain any drift down the chain.
        metadata = folder / "generation_metadata.json"
        if metadata.exists():
            quality = json.loads(metadata.read_text(encoding="utf-8"))["target_quality"]
            summary["parent_target_error_rate"] = quality["parent_argmax_error_rate_vs_true_answer"]
            summary["parent_out_of_context_rate"] = quality["parent_out_of_context_label_rate"]
            summary["distinct_questions"] = quality["distinct_questions"]
        summaries.append(summary)
    return summaries


def plot_generation_comparison(summaries, path):
    """Three panels: accuracy, circuit measures, and ablation effects by generation."""
    generations = [s["generation"] for s in summaries]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(generations, [s["final_measures"]["dev_acc"] for s in summaries],
                 marker="o", color="tab:blue")
    axes[0].axhline(0.2, color="grey", ls=":", lw=1, label="chance over 5 labels")
    axes[0].axhline(0.5, color="grey", ls="--", lw=1, label="chance within context")
    axes[0].set_ylabel("final development accuracy")
    axes[0].set_title("Accuracy by generation")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7)

    axes[1].plot(generations, [max(s["final_measures"]["induction_delta_per_head_layer1"])
                               for s in summaries], marker="o", label="strongest induction (L1)")
    axes[1].plot(generations, [max(s["final_measures"]["prev_token_label_positions_per_head_layer0"])
                               for s in summaries], marker="s", label="strongest previous-token (L0)")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("attention measure")
    axes[1].set_title("Circuit measures by generation")
    axes[1].legend(fontsize=7)

    # Ablation cost: how much accuracy each model loses when one head is silenced.
    for key, label in [("candidate_induction_head_ablated", "strongest induction head"),
                       ("candidate_previous_token_head_ablated", "strongest previous-token head")]:
        axes[2].plot(generations,
                     [s["ablations"]["intact"]["acc"] - s["ablations"][key]["acc"]
                      for s in summaries], marker="o", label=label)
    axes[2].axhline(0, color="black", lw=0.5)
    axes[2].set_ylabel("accuracy lost when the head is silenced")
    axes[2].set_title("Ablation cost by generation")
    axes[2].legend(fontsize=7)

    for axis in axes:
        axis.set_xlabel("generation")
        axis.set_xticks(generations)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def compare_generations(recursive_folder):
    """Write the across-generation comparison table and figure."""
    summaries = read_generation_summaries(recursive_folder)
    output = Path(recursive_folder)

    # Heads are chosen from each model's own final scores, never inherited. If
    # the same indices come out anyway, the per-generation numbers are directly
    # comparable; if they diverge, they are measuring different heads.
    head_keys = ["induction_layer1_head", "previous_token_layer0_head",
                 "comparison_layer1_head", "comparison_layer0_head"]
    heads_per_generation = {key: [s["candidate_heads"][key] for s in summaries]
                            for key in head_keys}
    heads_stable = {key: len(set(values)) == 1 for key, values in heads_per_generation.items()}

    rows = []
    for s in summaries:
        induction = s["final_measures"]["induction_delta_per_head_layer1"]
        previous = s["final_measures"]["prev_token_label_positions_per_head_layer0"]
        rows.append({
            "generation": s["generation"],
            "run": s["run"],
            "dev_accuracy": s["final_measures"]["dev_acc"],
            "dev_loss": s["final_measures"]["dev_loss"],
            "strongest_induction_score": max(induction),
            "strongest_previous_token_score": max(previous),
            "heads_with_positive_induction_score": sum(1 for v in induction if v > 0),
            "selected_heads": {key: s["candidate_heads"][key] for key in head_keys},
            "accuracy_lost_ablating_induction_head":
                s["ablations"]["intact"]["acc"] - s["ablations"]["candidate_induction_head_ablated"]["acc"],
            "accuracy_lost_ablating_previous_token_head":
                s["ablations"]["intact"]["acc"] - s["ablations"]["candidate_previous_token_head_ablated"]["acc"],
            "parent_target_error_rate": s.get("parent_target_error_rate"),
            "parent_out_of_context_rate": s.get("parent_out_of_context_rate"),
        })

    comparison = {
        "question": ("Does induction-circuit function weaken across recursive generations "
                     "before overall prediction accuracy declines?"),
        "chain": "one chain, generations 0 to {}".format(rows[-1]["generation"]),
        "evaluator": str(DEV_EVALUATOR_FILE.relative_to(ROOT)),
        "head_selection": ("each model's heads are chosen from its own final scores, never "
                           "inherited from another generation"),
        "selected_heads_identical_across_generations": heads_stable,
        "selected_heads_per_generation": heads_per_generation,
        "generations": rows,
    }
    (output / "generation_comparison.json").write_bytes(
        json.dumps(comparison, indent=2).encode("utf-8"))
    plot_generation_comparison(summaries, output / "generation_comparison.png")

    print("{:<5} {:>9} {:>9} {:>11} {:>11} {:>10} {:>10}".format(
        "gen", "dev acc", "dev loss", "induction", "prev-token", "abl IH", "abl PT"))
    for row in rows:
        print("{:<5} {:>9.4f} {:>9.4f} {:>11.4f} {:>11.4f} {:>10.4f} {:>10.4f}".format(
            row["generation"], row["dev_accuracy"], row["dev_loss"],
            row["strongest_induction_score"], row["strongest_previous_token_score"],
            row["accuracy_lost_ablating_induction_head"],
            row["accuracy_lost_ablating_previous_token_head"]))
    print("\nsame heads selected in every generation:", heads_stable)
    print("written:", (output / "generation_comparison.json").relative_to(ROOT))
    print("written:", (output / "generation_comparison.png").relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path, nargs="?",
                        help="A run folder to analyse")
    parser.add_argument("--compare", type=Path, default=None,
                        help="Folder holding generation_* runs; summarise them together "
                             "instead of analysing a single run")
    args = parser.parse_args()

    # Comparison mode only reads the per-run analyses, so every generation must
    # already have been analysed.
    if args.compare is not None:
        compare_generations(args.compare.resolve())
        return
    if args.run_folder is None:
        parser.error("give a run folder, or --compare with the folder holding the generations")
    folder = args.run_folder.resolve()
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    opts = common.load_run_options(folder)

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
