"""Analyse the extended-task generations and compare them.

Per generation: development performance at each output position, generated-symbol
behaviour, and the authors' previous-token and induction attention measures with
single-head ablations. Then one table and figure per condition, and one
comparison across conditions.

Generation 0 is shared, so it is analysed once and reused by every condition.

Results are discussed in findings/05_label_generation_strategies.md

Usage (from the project root):
  .venv/Scripts/python.exe scripts/extended_task/analyse_extended.py --condition label_argmax
  .venv/Scripts/python.exe scripts/extended_task/analyse_extended.py --compare-family label
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import extended_model as extended
import run_extended as pipeline

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The attention measures read the same positions as the base task: the
# previous-token score comes from layer 0, the induction score from layer 1 at
# the query symbol, which is token 4 of the seven-token extended sequence.
QUERY_TOKEN = extended.QUERY_LABEL_POSITION
CHECKPOINTS_TO_ANALYSE = 13


def choose_checkpoints(folder):
    """A fixed, roughly log-spaced subset of the saved checkpoints.

    The same rule is applied to every generation, so the analysed points line up
    across the chain. Returns (all saved, the subset analysed).
    """
    available = sorted(int(p.stem) for p in (folder / "checkpoints").glob("*.eqx"))
    targets = np.unique(np.concatenate(
        [[0], np.geomspace(max(available[1], 1), available[-1], CHECKPOINTS_TO_ANALYSE - 1)]))
    chosen = sorted({available[int(np.abs(np.asarray(available) - t).argmin())]
                     for t in targets})
    return available, chosen


def attention_measures(model, dev, key):
    """Previous-token and induction scores per head, using the authors' definitions.

    Previous-token score (layer 0): attention from each token to its predecessor,
    minus the chance share (1-a)/(1+r) where r indexes the shortened array. We
    average over the label tokens, positions 1 and 3.

    Induction score (layer 1): at the query symbol, attention to the label token
    following the matching context symbol minus attention to the other label
    token. Both are attention-pattern measurements, not causal quantities.
    """
    symbols = extended.add_next_symbol(dev["symbols"], dev["symbol_choice"])
    label_tokens = extended.add_label_slot(dev["labels"])
    keys = jax.random.split(key, symbols.shape[0])

    def one(symbol_row, label_row, row_key):
        aux = model.backbone.call_with_all_aux(examples=symbol_row, labels=label_row,
                                               key=row_key)
        return jnp.stack([block["attn_output"]["attn_scores"][0, :, :, :]
                          for block in aux["transformer_output"]["block_outputs"]])

    attention = np.asarray(jax.vmap(one)(symbols, label_tokens, keys))  # [n, 2, heads, 7, 7]
    sequence = attention.shape[3]
    rows = np.arange(1, sequence)
    raw = attention[:, 0, :, rows, rows - 1]            # [pair, batch, head]
    corrected = raw.transpose(0, 2, 1) - (1 - raw.transpose(0, 2, 1)) / (
        1 + np.arange(sequence - 1))[:, None, None]
    previous_token = corrected[0::2].mean(axis=(0, 2))   # label tokens only

    # Which context position holds the symbol matching the query.
    examples = np.asarray(dev["symbols"])
    correct = np.where(np.all(examples[:, 0] == examples[:, 2], axis=-1), 0, 1)
    query_row = attention[:, 1, :, QUERY_TOKEN, :]       # [batch, head, key]
    batch = np.arange(query_row.shape[0])
    induction = (query_row[batch, :, 2 * correct + 1]
                 - query_row[batch, :, 2 * (1 - correct) + 1]).mean(axis=0)
    return previous_token, induction


def generated_continuation_scores(model, dev, key):
    """Score the continuation the model generates for itself, autoregressively.

    Teacher forcing hands each position the correct earlier tokens; here the
    model conditions on what it actually produced, which is how a parent builds
    its successor's data. Decoding is the same in every condition — argmax
    labels, a temperature-1 symbol sample — because a condition's temperature
    belongs to data generation, not to evaluation.
    """
    symbol_key, label_key = jax.random.split(key)
    query, choice, following = extended.generate_continuation(
        model, dev["symbols"], dev["labels"][:, :2], symbol_key, label_key)
    true_next = jnp.take_along_axis(dev["labels"][:, :2], choice[:, None], axis=1)[:, 0]
    return {
        "generated_query_label_accuracy": float(jnp.mean(query == dev["labels"][:, 2])),
        "generated_next_label_accuracy": float(jnp.mean(following == true_next)),
        "generated_chose_context_position_0": float(jnp.mean(choice == 0)),
    }


def ablate(model, dev, layer, head, key):
    """Score the model with one head's value vectors zeroed.

    The block computes queries, keys and values with one fused projection whose
    output is read as (3, heads, head_dim). Zeroing the value rows for a single
    head makes that head write nothing into the residual stream, leaving every
    other head and weight untouched — the same intervention the base task used,
    applied here in weight space because this model is called directly.
    """
    attention = model.backbone.transformer.blocks[layer].attn
    width = attention.qkv.weight.shape[1]                 # model dimension
    head_dim = width // attention.num_heads
    start = 2 * width + head * head_dim                   # 2 = the value third
    weight = attention.qkv.weight.at[start:start + head_dim, :].set(0.0)
    patched = eqx.tree_at(
        lambda m: m.backbone.transformer.blocks[layer].attn.qkv.weight, model, weight)
    if attention.qkv.bias is not None:
        bias = attention.qkv.bias.at[start:start + head_dim].set(0.0)
        patched = eqx.tree_at(
            lambda m: m.backbone.transformer.blocks[layer].attn.qkv.bias, patched, bias)
    return pipeline.score_dev(patched, dev, key)


def analyse_generation(folder, opts, dev):
    """Measure one generation and write its analysis record."""
    available, chosen = choose_checkpoints(folder)
    key = jax.random.PRNGKey(0)

    trajectory = []
    for iteration in chosen:
        model = pipeline.load_model(folder, iteration, opts)
        previous_token, induction = attention_measures(model, dev, key)
        scores = pipeline.score_dev(model, dev, key)
        trajectory.append({
            "iter": int(iteration),
            "query_label_accuracy": scores["query_label_accuracy"],
            "symbol_loss": scores["symbol_loss"],
            "next_label_accuracy": scores["next_label_accuracy"],
            "max_previous_token_score": float(previous_token.max()),
            "max_induction_score": float(induction.max()),
        })

    final = pipeline.load_model(folder, available[-1], opts)
    previous_token, induction = attention_measures(final, dev, key)
    scores = pipeline.score_dev(final, dev, key)
    generated = generated_continuation_scores(final, dev, key)

    # Heads are picked from this model's own scores, never inherited.
    induction_head = int(np.argmax(induction))
    previous_head = int(np.argmax(previous_token))
    intact = scores["query_label_accuracy"]
    without_induction = ablate(final, dev, 1, induction_head, key)["query_label_accuracy"]
    without_previous = ablate(final, dev, 0, previous_head, key)["query_label_accuracy"]
    ablations = {
        "intact_query_label_accuracy": intact,
        # Head identities are recorded because each model picks its own: this is a
        # descriptive comparison, not one fixed head tracked across models.
        "induction_head": induction_head,
        "previous_token_head": previous_head,
        "query_accuracy_without_induction_head": without_induction,
        "query_accuracy_without_previous_token_head": without_previous,
        "induction_head_ablation_effect": intact - without_induction,
        "previous_token_head_ablation_effect": intact - without_previous,
    }

    quality = json.loads((folder / "dataset_quality.json").read_text(encoding="utf-8"))
    summary = {
        "run": str(folder.relative_to(common.ROOT)),
        "generation": int(json.loads((folder / "config.json").read_text())["generation"]),
        "checkpoints_saved": len(available),
        "checkpoints_analysed": [int(i) for i in chosen],
        "final_dev_scores": scores,
        "final_generated_scores": generated,
        "trajectory": trajectory,
        "previous_token_scores_layer0": previous_token.tolist(),
        "induction_scores_layer1": induction.tolist(),
        "ablations": ablations,
        "training_data_quality": quality,
        "note": ("final_dev_scores are teacher forced: each position is conditioned on "
                 "the correct earlier continuation tokens. final_generated_scores let the "
                 "model condition on its own output instead. training_data_quality "
                 "describes the generated data this model was trained on."),
    }
    (folder / "analysis.json").write_bytes(json.dumps(summary, indent=2).encode("utf-8"))
    return summary


def label_breakdown(generated, correct, context_labels):
    """Where a generated label went, judged against the original context mapping.

    The two context labels always differ in these datasets, so the three
    outcomes are exclusive and exhaustive: the correct label, the other label
    present in the context, or a label from outside the context entirely.

    generated, correct: [n]. context_labels: [n, 2].
    Returns counts, proportions, both label histograms, and the 5x5 table of
    correct label against generated label.
    """
    other = np.where(context_labels[:, 0] == correct, context_labels[:, 1],
                     context_labels[:, 0])
    is_correct = generated == correct
    is_other = (~is_correct) & (generated == other)
    total = len(generated)
    table = np.zeros((5, 5), np.int64)
    np.add.at(table, (correct.astype(np.int64), generated.astype(np.int64)), 1)
    return {
        "correct": int(is_correct.sum()),
        "other_context_label": int(is_other.sum()),
        "outside_context_labels": int(total - is_correct.sum() - is_other.sum()),
        "correct_proportion": float(is_correct.mean()),
        "other_context_label_proportion": float(is_other.mean()),
        "outside_context_labels_proportion": float(1 - is_correct.mean() - is_other.mean()),
        "generated_label_counts": np.bincount(generated.astype(np.int64), minlength=5).tolist(),
        "correct_label_counts": np.bincount(correct.astype(np.int64), minlength=5).tolist(),
        # Row = correct label, column = generated label.
        "correct_vs_generated_counts": table.tolist(),
    }


def symbol_breakdown(class_idxs, symbol_choice, class_count):
    """How often each training class was offered as a next symbol, and then chosen.

    Both context slots are filled by the question generator, so a class chosen
    often may simply have been offered often. `selection_rate` divides chosen by
    offered; the intended coin flip gives 0.5 for every class. The per-class
    arrays are kept at full length so a class that stopped being offered shows as
    a zero rather than disappearing. Counting offers this way assumes the two
    context classes differ, which holds in every row of these datasets.
    """
    chosen_class = class_idxs[np.arange(len(symbol_choice)), symbol_choice]
    offered = (np.bincount(class_idxs[:, 0].astype(np.int64), minlength=class_count)
               + np.bincount(class_idxs[:, 1].astype(np.int64), minlength=class_count)
               )[:class_count]
    chosen = np.bincount(chosen_class.astype(np.int64), minlength=class_count)[:class_count]
    offered_share = offered / offered.sum()
    share = chosen / chosen.sum()
    seen = offered > 0
    rate = np.where(seen, chosen / np.maximum(offered, 1), np.nan)
    present = share[share > 0]
    return {
        "distinct_classes_offered": int(seen.sum()),
        "distinct_classes_selected": int(np.sum(chosen > 0)),
        "offered_counts_per_class": offered.tolist(),
        "chosen_counts_per_class": chosen.tolist(),
        "selection_rate_per_class": [None if np.isnan(v) else float(v) for v in rate],
        "largest_offered_share": float(offered_share.max()),
        "largest_class_share": float(share.max()),
        # Availability-adjusted identity preference: 0.5 everywhere means the
        # parent picked between the two offered symbols without favouring any.
        "selection_rate_min": float(np.nanmin(rate)),
        "selection_rate_max": float(np.nanmax(rate)),
        "selection_rate_sd": float(np.nanstd(rate)),
        "coin_flip_sd_reference": float(np.sqrt(0.25 / offered[seen].mean())),
        "chosen_class_entropy_nats": float(-np.sum(present * np.log(present))),
        "offered_class_entropy_nats": float(
            -np.sum(offered_share[offered_share > 0] * np.log(offered_share[offered_share > 0]))),
        "max_possible_entropy_nats": float(np.log(class_count)),
    }


def dataset_distributions(folder):
    """What a parent actually generated, read from that generation's saved dataset.

    Symbols and labels are reported separately: which symbol identities were
    chosen once availability is accounted for, and where each of the two label
    outputs went. Generation 0's dataset holds correct continuations and serves
    as the reference.

    Under context feedback four stages have to be kept apart, and this record
    carries all four: the frequencies the parent generated and the sampling
    weights derived from them (both taken from the run's `config.json`), then the
    symbols actually offered in the child's contexts and the ones it selected
    (both counted here).
    """
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    with h5py.File(folder / "training_data.h5", "r") as handle:
        class_idxs = handle["class_idxs"][:]
        labels = handle["labels"][:]
        symbol_choice = handle["symbol_choice"][:].astype(np.int64)
        next_label = handle["next_label"][:]
        true_query = handle["true_query_label"][:]
        true_next = handle["true_next_label"][:]
    feedback = config.get("context_feedback")
    return {
        "generation": int(config["generation"]),
        "examples": int(len(symbol_choice)),
        "next_symbol_temperature": config.get("next_symbol_temperature", 1.0),
        "context_selection_temperature": config.get("context_selection_temperature"),
        # Stages 1 and 2 of context feedback; absent when the questions came from
        # the original task distribution.
        "parent_generated_frequencies": None if feedback is None
        else feedback["parent_generated_frequencies"],
        "context_sampling_weights": None if feedback is None else feedback["sampling_weights"],
        "chose_context_position_0": float(np.mean(symbol_choice == 0)),
        # Stages 3 and 4: what was offered, and what was chosen.
        "next_symbol": symbol_breakdown(class_idxs, symbol_choice,
                                        int(config["class_split"][0])),
        # labels[:, 2] is the generated query label; labels[:, :2] the context pair.
        "query_label": label_breakdown(labels[:, 2], true_query, labels[:, :2]),
        "following_label": label_breakdown(next_label, true_next, labels[:, :2]),
    }


def plot_chain(summaries, path):
    """Four panels: performance, symbol behaviour, attention measures, learning curves."""
    generations = [s["generation"] for s in summaries]
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    axes[0].plot(generations, [s["final_dev_scores"]["query_label_accuracy"] for s in summaries],
                 marker="o", label="query label")
    axes[0].plot(generations, [s["final_dev_scores"]["next_label_accuracy"] for s in summaries],
                 marker="s", label="following label")
    axes[0].axhline(0.2, color="grey", ls=":", lw=1, label="chance over 5 labels")
    axes[0].set_ylabel("teacher-forced development accuracy")
    axes[0].set_title("Label performance by generation")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(generations,
                 [s["training_data_quality"]["chose_context_position_0"] for s in summaries],
                 marker="o", label="generated: chose position 0")
    axes[1].axhline(0.5, color="grey", ls="--", lw=1, label="intended 0.5")
    axes[1].set_ylabel("proportion")
    axes[1].set_title("Generated next-symbol choice")
    axes[1].set_ylim(0, 1.05)

    axes[2].plot(generations, [max(s["induction_scores_layer1"]) for s in summaries],
                 marker="o", label="strongest induction (L1)")
    axes[2].plot(generations, [max(s["previous_token_scores_layer0"]) for s in summaries],
                 marker="s", label="strongest previous-token (L0)")
    axes[2].axhline(0, color="black", lw=0.5)
    axes[2].set_ylabel("attention measure")
    axes[2].set_title("Attention measures by generation")

    # Within-training curves, from the metrics each run logged every 5,000 sequences.
    for summary in summaries:
        with h5py.File(common.ROOT / summary["run"] / "log.h5", "r") as handle:
            axes[3].plot(handle["eval_iter"][:], handle["dev/query_label_accuracy"][:],
                         lw=1.2, label="generation {}".format(summary["generation"]))
    axes[3].axhline(0.2, color="grey", ls=":", lw=1)
    axes[3].set_xlabel("training sequences")
    axes[3].set_ylabel("teacher-forced query-label accuracy")
    axes[3].set_title("Within-training curves")
    axes[3].set_ylim(0, 1.05)
    axes[3].legend(fontsize=7)

    for axis in axes[:3]:
        axis.set_xlabel("generation")
        axis.set_xticks(generations)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def chain_folders(condition):
    """The shared generation 0 followed by this condition's successors."""
    # Only numbered directories: the folder also holds the comparison outputs.
    successors = [f for f in (pipeline.EXPERIMENTS / condition).glob("generation_*")
                  if f.is_dir() and f.name.split("_")[1].isdigit()]
    return [pipeline.GENERATION_0] + sorted(successors,
                                            key=lambda f: int(f.name.split("_")[1]))


def summarise_condition(condition, opts, dev):
    """Analyse every generation in one condition and write its table and figure.

    A generation that already has an `analysis.json` is reused rather than
    recomputed, so switching folders or adding a condition never re-measures
    work that is already done.
    """
    summaries, label_rule, distributions = [], None, []
    for folder in chain_folders(condition):
        record = folder / "analysis.json"
        if record.exists() and "final_generated_scores" in json.loads(
                record.read_text(encoding="utf-8")):
            summary = json.loads(record.read_text(encoding="utf-8"))
            summary["run"] = str(folder.relative_to(common.ROOT))
            # Read the quality record fresh: it is the source of truth, and the
            # cached copy inside an older analysis may predate added fields.
            summary["training_data_quality"] = json.loads(
                (folder / "dataset_quality.json").read_text(encoding="utf-8"))
        else:
            summary = analyse_generation(folder, opts, dev)
        summaries.append(summary)

        # Read straight from the saved dataset, so this never depends on a
        # cached record. Generation 0 is shared, so its distributions are
        # written once into its own folder rather than into every condition.
        record = dataset_distributions(folder)
        if record["generation"] == 0:
            (folder / "dataset_distributions.json").write_bytes(
                json.dumps(record, indent=2).encode("utf-8"))
        else:
            distributions.append(record)
        if summary["generation"] > 0:
            label_rule = json.loads(
                (folder / "config.json").read_text(encoding="utf-8"))["label_rule"]
        print("generation {}: query acc {:.4f}  next acc {:.4f}  symbol loss {:.4f}".format(
            summary["generation"], summary["final_dev_scores"]["query_label_accuracy"],
            summary["final_dev_scores"]["next_label_accuracy"],
            summary["final_dev_scores"]["symbol_loss"]), flush=True)

    output = pipeline.EXPERIMENTS / condition
    comparison = {
        "condition": condition,
        "label_generation": label_rule,
        "question": ("Does query-label performance, generated-symbol behaviour or the "
                     "attention measures change across extended-task generations?"),
        "evaluator": str(common.DEV_EVALUATOR_FILE.relative_to(common.ROOT)),
        "evaluation": ("teacher forced, identical for every condition; generated_continuation "
                       "instead lets the model condition on its own output, and the "
                       "training_data_quality columns describe the data the parent produced"),
        "head_selection": "each model's heads chosen from its own final scores",
        "checkpoints_saved_per_generation": summaries[0]["checkpoints_saved"],
        "checkpoints_analysed_per_generation": summaries[0]["checkpoints_analysed"],
        "generations": [condition_row(s) for s in summaries],
    }
    (output / "generation_comparison.json").write_bytes(
        json.dumps(comparison, indent=2).encode("utf-8"))
    plot_chain(summaries, output / "generation_comparison.png")
    (output / "dataset_distributions.json").write_bytes(json.dumps({
        "condition": condition,
        "label_generation": label_rule,
        "source": "counted from each generation's saved training_data.h5",
        "note": ("what the parent generated, against the original context mapping. "
                 "The dataset for generation n was produced by generation n-1 and "
                 "used to train generation n. Generation 0's own dataset holds "
                 "correct continuations and is recorded beside it, in "
                 "results/extended_task/recursive/generation_0/"),
        "generations": distributions,
    }, indent=2).encode("utf-8"))
    print("written:", (output / "generation_comparison.json").relative_to(common.ROOT))
    print("written:", (output / "dataset_distributions.json").relative_to(common.ROOT))
    return comparison


def condition_row(summary):
    """One generation's headline numbers, the same fields for every condition."""
    scores = summary["final_dev_scores"]
    return {
        "generation": summary["generation"],
        # Teacher forced: every position sees the correct earlier tokens.
        "query_label_accuracy": scores["query_label_accuracy"],
        "query_label_loss": scores["query_label_loss"],
        "symbol_loss": scores["symbol_loss"],
        "symbol_prefers_first": scores["symbol_prefers_first"],
        "next_label_accuracy": scores["next_label_accuracy"],
        "next_label_loss": scores["next_label_loss"],
        # Generated: the model conditions on its own earlier output.
        "generated_continuation": summary["final_generated_scores"],
        "strongest_induction_score": max(summary["induction_scores_layer1"]),
        "strongest_previous_token_score": max(summary["previous_token_scores_layer0"]),
        "ablations": summary["ablations"],
        "training_data_quality": summary["training_data_quality"],
    }


def plot_conditions(comparisons, path):
    """Every condition on shared axes, so the trajectories can be compared.

    Six panels: teacher-forced accuracy and loss, the corruption of the training
    targets, the two attention measures, and the measured effect of removing each
    model's strongest induction head.
    """
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.4))
    axes = axes.ravel()
    for comparison in comparisons:
        rows = comparison["generations"]
        generations = [r["generation"] for r in rows]
        label = comparison["condition"]
        axes[0].plot(generations, [r["query_label_accuracy"] for r in rows],
                     marker="o", label=label)
        axes[1].plot(generations, [r["query_label_loss"] for r in rows],
                     marker="o", label=label)
        # Generation 0 trains on correct data, so it has no generated-target errors.
        axes[2].plot(generations[1:],
                     [r["training_data_quality"]["query_label_error_rate"] for r in rows[1:]],
                     marker="o", label=label)
        axes[3].plot(generations, [r["strongest_induction_score"] for r in rows],
                     marker="o", label=label)
        axes[4].plot(generations, [r["strongest_previous_token_score"] for r in rows],
                     marker="o", label=label)
        axes[5].plot(generations,
                     [r["ablations"]["induction_head_ablation_effect"] for r in rows],
                     marker="o", label=label)

    axes[0].axhline(0.2, color="grey", ls=":", lw=1, label="chance over 5 labels")
    axes[0].set_ylabel("teacher-forced query-label accuracy")
    axes[0].set_title("Query-label accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("query-label loss (nats)")
    axes[1].set_title("Query-label loss")
    axes[2].set_ylabel("wrong query labels in the training data")
    axes[2].set_title("Corruption of the generated targets")
    axes[3].set_ylabel("strongest induction score (L1)")
    axes[3].set_title("Induction attention measure")
    axes[4].set_ylabel("strongest previous-token score (L0)")
    axes[4].set_title("Previous-token attention measure")
    axes[5].axhline(0, color="black", lw=0.5)
    axes[5].set_ylabel("accuracy lost when the head is silenced")
    axes[5].set_title("Induction-head ablation effect")
    for axis in axes:
        axis.set_xlabel("generation")
        axis.set_xticks([r["generation"] for r in comparisons[0]["generations"]])
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_distributions(records, reference, path):
    """Three panels on what the parents generated, across conditions.

    `records` maps condition name to its list of per-generation distribution
    records; `reference` is generation 0's, whose data is correct by
    construction. Left: generated query labels at the last generation against
    the true labels for the same questions. Middle: of the query answers that
    were wrong, how many left the context entirely — drawn only where at least
    1,000 answers were wrong, because the near-perfect conditions would otherwise
    show a ratio taken over a handful of examples. Right: how far per-class
    symbol selection strayed from the intended coin flip.
    """
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    labels = np.arange(5)

    # Left: the true frequencies are not uniform, so they are the comparison.
    width = 0.8 / (len(records) + 1)
    true_counts = np.asarray(reference["query_label"]["correct_label_counts"], float)
    axes[0].bar(labels - 0.4 + width / 2, true_counts / true_counts.sum(), width,
                label="true labels", color="black")
    for index, (condition, generations) in enumerate(sorted(records.items())):
        counts = np.asarray(generations[-1]["query_label"]["generated_label_counts"], float)
        axes[0].bar(labels - 0.4 + width * (index + 1.5), counts / counts.sum(), width,
                    label=condition)
    axes[0].set_xticks(labels)
    axes[0].set_xlabel("label")
    axes[0].set_ylabel("proportion of the million examples")
    axes[0].set_title("Generated query labels, final generation")

    for condition, generations in sorted(records.items()):
        gens = [g["generation"] for g in generations]
        # Of the wrong answers, the share that was not even a context label.
        outside = []
        for g in generations:
            wrong = g["query_label"]["other_context_label"] +                 g["query_label"]["outside_context_labels"]
            outside.append(g["query_label"]["outside_context_labels"] / wrong
                           if wrong >= 1000 else float("nan"))
        axes[1].plot(gens, outside, marker="o", label=condition)
        axes[2].plot(gens, [g["next_symbol"]["selection_rate_sd"] for g in generations],
                     marker="o", label=condition)

    axes[1].axhline(0.75, color="grey", ls=":", lw=1,
                    label="uniform over 5 labels (3 of 4 wrong answers)")
    axes[1].set_ylabel("share of wrong query labels")
    axes[1].set_title("Wrong answers that left the context")
    axes[1].set_ylim(0, 1.05)
    axes[2].axhline(reference["next_symbol"]["selection_rate_sd"], color="black", ls="--",
                    lw=1, label="generation 0 (correct data)")
    axes[2].axhline(reference["next_symbol"]["coin_flip_sd_reference"], color="grey", ls=":",
                    lw=1, label="coin-flip sampling spread")
    axes[2].set_ylabel("sd of per-class selection rate")
    axes[2].set_title("Symbol-identity preference, availability adjusted")
    for axis in axes[1:]:
        axis.set_xlabel("generation")
        axis.set_xticks(sorted({g["generation"] for gens in records.values()
                                for g in gens}))
    for axis in axes:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# Each family compares the shared reference condition against the conditions that
# change one thing from it. The reference appears in all three.
REFERENCE_CONDITION = "label_argmax"
FAMILIES = {
    "label": ("label_sampling_temperature_", "condition_comparison"),
    "context": ("context_feedback_temperature_", "context_feedback_comparison"),
    "symbol": ("symbol_sampling_temperature_", "symbol_sampling_comparison"),
}


def family_conditions(family):
    """The reference condition followed by that family's conditions, in folder order."""
    prefix = FAMILIES[family][0]
    others = sorted(f.name for f in pipeline.EXPERIMENTS.glob(prefix + "*") if f.is_dir())
    return [REFERENCE_CONDITION] + others


def plot_symbol_distributions(records, reference, path):
    """Four panels on the symbol side of the generated data.

    `records` maps condition name to its per-generation distribution records.
    Concentration separates the stages that context feedback introduces: the
    weights supplied to the question generator, the symbols actually offered, and
    the symbols the parent then selected. Position preference and identity
    preference are kept apart, because colder sampling could move either.
    """
    figure, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for condition, generations in sorted(records.items()):
        gens = [g["generation"] for g in generations]
        axes[0].plot(gens, [g["next_symbol"]["largest_class_share"] for g in generations],
                     marker="o", label=condition)
        if generations[0]["context_sampling_weights"] is not None:
            axes[0].plot(gens, [max(g["context_sampling_weights"]) for g in generations],
                         ls=":", marker="x", label=condition + " (weights supplied)")
            axes[0].plot(gens, [g["next_symbol"]["largest_offered_share"]
                                for g in generations],
                         ls="--", marker="s", label=condition + " (offered)")
        axes[1].plot(gens, [g["next_symbol"]["distinct_classes_selected"]
                            for g in generations], marker="o", label=condition)
        axes[2].plot(gens, [g["chose_context_position_0"] for g in generations],
                     marker="o", label=condition)
        axes[3].plot(gens, [g["next_symbol"]["selection_rate_sd"] for g in generations],
                     marker="o", label=condition)

    axes[0].axhline(reference["next_symbol"]["largest_class_share"], color="black", ls="--",
                    lw=1, label="generation 0 (correct data)")
    axes[0].set_ylabel("largest single-class share")
    axes[0].set_title("Symbol concentration, by stage")
    axes[1].axhline(reference["next_symbol"]["distinct_classes_selected"], color="black",
                    ls="--", lw=1, label="all 50 classes")
    axes[1].set_ylabel("distinct classes selected")
    axes[1].set_title("Symbol coverage")
    axes[2].axhline(0.5, color="grey", ls=":", lw=1, label="even split")
    axes[2].set_ylabel("proportion choosing context position 0")
    axes[2].set_title("Positional preference")
    axes[3].axhline(reference["next_symbol"]["coin_flip_sd_reference"], color="grey", ls=":",
                    lw=1, label="coin-flip sampling spread")
    axes[3].set_ylabel("sd of per-class selection rate")
    axes[3].set_title("Identity preference, availability adjusted")
    for axis in axes:
        axis.set_xlabel("generation")
        axis.set_xticks(sorted({g["generation"] for gens in records.values()
                                for g in gens}))
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", default=None,
                        help="Folder name under results/extended_task/experiments/")
    parser.add_argument("--compare-family", choices=sorted(FAMILIES),
                        help="Compare the shared reference condition against one family: "
                             "label generation, context feedback or next-symbol temperature")
    args = parser.parse_args()

    opts = common.baseline_options()
    opts.model_output_classes = opts.fs_relabel

    # Comparing a family only reads tables the per-condition runs already wrote.
    if args.compare_family:
        conditions = family_conditions(args.compare_family)
        stem = FAMILIES[args.compare_family][1]
        comparisons = []
        for name in conditions:
            table = pipeline.EXPERIMENTS / name / "generation_comparison.json"
            if table.exists():
                comparisons.append(json.loads(table.read_text(encoding="utf-8")))
        if len(comparisons) < 2:
            raise SystemExit("Need the reference plus at least one analysed condition.")
        (pipeline.RECURSIVE / (stem + ".json")).write_bytes(json.dumps({
            "question": ("How does the synthetic-label generation strategy affect "
                         "performance degradation and induction-circuit function "
                         "across recursive generations?"),
            "shared_generation_0": str(pipeline.GENERATION_0.relative_to(common.ROOT)),
            "evaluation": "identical fixed development questions and decoding for both conditions",
            "family": args.compare_family,
            "reference_condition": REFERENCE_CONDITION,
            "note": ("every condition changes one thing from the reference, which uses the "
                     "original opening questions, argmax labels and next-symbol "
                     "temperature 1. One chain per condition from a common parent, not "
                     "independent replications"),
            "conditions": comparisons,
        }, indent=2).encode("utf-8"))
        plot_conditions(comparisons, pipeline.RECURSIVE / (stem + ".png"))

        # The distribution figures read the records each condition already wrote.
        records = {}
        for name in conditions:
            table = pipeline.EXPERIMENTS / name / "dataset_distributions.json"
            if table.exists():
                records[name] = json.loads(table.read_text(encoding="utf-8"))["generations"]
        reference = json.loads((pipeline.GENERATION_0 / "dataset_distributions.json")
                               .read_text(encoding="utf-8"))
        if args.compare_family == "label":
            # The label family's question is where the generated labels went.
            figure_path = pipeline.RECURSIVE / "data_distributions.png"
            plot_distributions(records, reference, figure_path)
        else:
            # The symbol families ask about concentration, coverage, position
            # preference and identity preference instead.
            figure_path = pipeline.RECURSIVE / (
                stem.replace("comparison", "distributions") + ".png")
            plot_symbol_distributions(records, reference, figure_path)
        print("written:", figure_path.relative_to(common.ROOT))
        for comparison in comparisons:
            print("\n" + comparison["condition"])
            for row in comparison["generations"]:
                quality = row["training_data_quality"]
                print("  gen {}: query acc {:.4f}  induction {:.4f}  wrong query labels {}".format(
                    row["generation"], row["query_label_accuracy"],
                    row["strongest_induction_score"],
                    quality.get("query_label_errors", 0) if row["generation"] else "-"))
        print("\nwritten:", (pipeline.RECURSIVE / (stem + ".json")).relative_to(common.ROOT))
        print("written:", (pipeline.RECURSIVE / (stem + ".png")).relative_to(common.ROOT))
        return

    if args.condition is None:
        parser.error("give --condition, or --compare-family")
    summarise_condition(args.condition, opts, pipeline.load_dev_set(opts))


if __name__ == "__main__":
    main()
