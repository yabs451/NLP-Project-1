"""Learning-rate search for the extended task.

Trains a 6 x 3 grid (six learning rates, three initialisation seeds) on the
original extended task with correct labels, and scores each final checkpoint on
the fixed 1,000-question development evaluator. Selection uses query-label
accuracy only; the symbol and following-label numbers are recorded alongside it
for interpretation.

Method and results: findings/04_extended_task_tuning.md

Usage (from the project root):
  .venv/Scripts/python.exe scripts/extended_task/tune_extended.py
  .venv/Scripts/python.exe scripts/extended_task/tune_extended.py --dry-run
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import run_extended as pipeline

import h5py
import jax
import numpy as np
import main_utils

LEARNING_RATES = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.001]
INIT_SEEDS = [5, 6, 7]
PUBLISHED_LEARNING_RATE = 0.00001
# Seed 5 is fixed in advance as the seed carried into the recursive chain. It is
# not whichever seed happens to score best.
GENERATION_0_SEED = 5

# Fixed in the code before any result was inspected, and not revised afterwards.
SELECTION_RULE = ("highest unrounded mean final query-label development accuracy over "
                  "init seeds 5, 6 and 7; exact ties broken by lowest mean final "
                  "query-label loss, then by preferring 1e-05, then the smaller rate; "
                  "a rate needs all three seeds completed to be eligible")

TUNING = pipeline.TUNING
RESULTS_FILE = TUNING / "results.json"
SELECTION_FILE = pipeline.SELECTION_FILE
FIGURE_FILE = TUNING / "learning_rate_comparison.png"
# Every candidate trains on the same original-task data, so it is built once.
SHARED_DATA_FILE = TUNING / "training_data.h5"

# Settings a candidate must share for its score to be comparable. Checked against
# a run's own config.json before that run is reused or accepted.
SHARED_SETTINGS = ["train_seed", "eval_seed", "train_iters", "train_bs", "optimizer",
                   "weight_decay", "d_model", "num_heads", "depth", "mlp_ratio",
                   "class_split", "exemplar_split", "fs_relabel", "fs_relabel_split_seed",
                   "train_context_len", "noise_scale_train", "task",
                   "symbol_head_init_seed", "training_choice_seed", "dev_choice_seed"]


def candidate_name(learning_rate, init_seed):
    """Folder name stating the two settings that vary. '1e-06' means 0.000001."""
    return "learning_rate_{:g}_init_seed_{}".format(learning_rate, init_seed)


def load_shared_dataset(opts, splits, reference):
    """The original extended-task dataset, built once and reused by every candidate.

    It depends on neither the learning rate nor the initialisation seed: the
    opening questions come from the training-data seed and the next-symbol
    targets from a fixed coin-flip seed. One copy therefore serves all eighteen
    candidates. A matching generation-0 run already holds exactly this dataset,
    so it is read from there rather than rebuilt.
    """
    existing = pipeline.GENERATION_0 / "training_data.h5"
    for source in (SHARED_DATA_FILE, existing):
        if source == existing and not configuration_matches(
                pipeline.GENERATION_0, PUBLISHED_LEARNING_RATE, GENERATION_0_SEED,
                reference)[0]:
            continue
        if source.exists():
            print("reading the original-task dataset from",
                  source.relative_to(common.ROOT), flush=True)
            with h5py.File(source, "r") as handle:
                return {name: handle[name][:] for name in handle}
    print("building the original-task dataset", flush=True)
    dataset = pipeline.build_generation_0_dataset(opts, splits, opts.train_iters)
    pipeline.save_dataset(SHARED_DATA_FILE, dataset)
    return dataset


def configuration_matches(run_folder, learning_rate, init_seed, reference):
    """Check a finished run really has the configuration this candidate needs.

    Returns (matches, reason). A successor is rejected: its config.json looks
    similar, but its targets came from a parent model rather than from the task.
    """
    config_file = Path(run_folder) / "config.json"
    if not config_file.exists():
        return False, "no config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if int(config.get("generation", -1)) != 0:
        return False, "trained on generated targets, not the original task"
    if float(config.get("lr", float("nan"))) != learning_rate:
        return False, "learning rate is {}".format(config.get("lr"))
    if int(config.get("init_seed", -1)) != init_seed:
        return False, "init seed is {}".format(config.get("init_seed"))
    for setting in SHARED_SETTINGS:
        if config.get(setting) != reference.get(setting):
            return False, "{} is {}, expected {}".format(setting, config.get(setting),
                                                         reference.get(setting))
    return True, "configuration matches"


def is_complete(run_folder, expected_iterations):
    """A run counts as finished only if its final checkpoint is actually there."""
    if not (Path(run_folder) / "checkpoints").is_dir():
        return False
    return expected_iterations in common.available_checkpoints(run_folder)


def find_reusable_run(learning_rate, init_seed, reference, expected_iterations):
    """An already-finished generation-0 run with this configuration, if one exists.

    Looks at the shared generation 0 as well as the tuning folder, so an existing
    matching model is scored rather than retrained.
    """
    for folder in [pipeline.GENERATION_0] + sorted(TUNING.glob("learning_rate_*")):
        if not folder.is_dir():
            continue
        matches, _ = configuration_matches(folder, learning_rate, init_seed, reference)
        if matches and is_complete(folder, expected_iterations):
            return folder
    return None


def train_candidate(folder, opts, learning_rate, init_seed, dataset, features, dev):
    """Train one candidate, keeping only the first and last checkpoint.

    Tuning compares final models, so the mechanistic 55-point schedule is not
    used here.
    """
    candidate_opts = argparse.Namespace(**vars(opts))
    candidate_opts.lr = learning_rate
    candidate_opts.init_seed = init_seed
    folder.mkdir(parents=True, exist_ok=True)
    pipeline.write_config(folder, candidate_opts, 0, None, "argmax", None)
    pipeline.train_generation(candidate_opts, folder, dataset, features, dev,
                              checkpoints=[0, candidate_opts.train_iters])


def score_final_checkpoint(run_folder, opts, learning_rate, init_seed, dev):
    """Teacher-forced development scores of one candidate's final model.

    Returns the full score dictionary: query-label accuracy and loss decide the
    selection, the symbol and following-label numbers are kept for interpretation.
    """
    candidate_opts = argparse.Namespace(**vars(opts))
    candidate_opts.lr = learning_rate
    candidate_opts.init_seed = init_seed
    model = pipeline.load_final_model(run_folder, candidate_opts)
    return pipeline.score_dev(model, dev, jax.random.PRNGKey(0))


def summarise_by_learning_rate(records):
    """Average each learning rate over the seeds that completed, in grid order."""
    summary = []
    for learning_rate in LEARNING_RATES:
        done = {seed: records[(learning_rate, seed)] for seed in INIT_SEEDS
                if (learning_rate, seed) in records
                and records[(learning_rate, seed)]["status"] == "completed"}
        accuracies = [row["dev_query_label_accuracy"] for row in done.values()]
        losses = [row["dev_query_label_loss"] for row in done.values()]
        summary.append({
            "learning_rate": learning_rate,
            "seeds_scored": sorted(done),
            "per_seed_query_accuracy": {s: done[s]["dev_query_label_accuracy"]
                                        for s in sorted(done)},
            "mean_query_accuracy": float(np.mean(accuracies)) if accuracies else None,
            "std_query_accuracy": float(np.std(accuracies)) if accuracies else None,
            "mean_query_loss": float(np.mean(losses)) if losses else None,
            # Recorded for interpretation only; they do not enter the selection.
            "mean_symbol_loss": (float(np.mean([r["dev_symbol_loss"] for r in done.values()]))
                                 if done else None),
            "mean_next_label_accuracy":
                (float(np.mean([r["dev_next_label_accuracy"] for r in done.values()]))
                 if done else None),
        })
    return summary


def select_learning_rate(summary):
    """Apply SELECTION_RULE to the per-rate summary, or return None if nothing is eligible."""
    eligible = [row for row in summary if len(row["seeds_scored"]) == len(INIT_SEEDS)]
    if not eligible:
        return None
    best = max(row["mean_query_accuracy"] for row in eligible)
    tied = [row for row in eligible if row["mean_query_accuracy"] == best]
    if len(tied) > 1:
        lowest = min(row["mean_query_loss"] for row in tied)
        tied = [row for row in tied if row["mean_query_loss"] == lowest]
    if len(tied) > 1:
        published = [row for row in tied if row["learning_rate"] == PUBLISHED_LEARNING_RATE]
        tied = published or [min(tied, key=lambda row: row["learning_rate"])]
    return tied[0]


def plot_summary(summary, winner, path):
    """Final query-label accuracy against learning rate, with the individual seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for row in summary:
        for accuracy in row["per_seed_query_accuracy"].values():
            axis.scatter(row["learning_rate"], accuracy, s=24, color="tab:blue",
                         alpha=0.55, zorder=3)
    rates = [r["learning_rate"] for r in summary if r["mean_query_accuracy"] is not None]
    means = [r["mean_query_accuracy"] for r in summary if r["mean_query_accuracy"] is not None]
    axis.plot(rates, means, color="tab:blue", marker="o", markersize=5,
              label="mean of seeds 5, 6, 7", zorder=2)
    if winner is not None:
        axis.axvline(winner["learning_rate"], color="tab:red", ls="--", alpha=0.6,
                     label="selected: {:g}".format(winner["learning_rate"]))
    axis.axhline(0.2, color="grey", ls=":", lw=1, label="chance over 5 labels (20%)")
    axis.axhline(0.5, color="grey", ls="--", lw=1, label="chance within context (50%)")
    axis.set_xscale("log")
    axis.set_xlabel("constant Adam learning rate (log scale)")
    axis.set_ylabel("final query-label accuracy, 1,000 development questions")
    axis.set_title("Extended task: learning-rate search after 31,250 updates")
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def load_records():
    """Previous grid results, keyed by (rate, seed) so a retry replaces its entry."""
    if RESULTS_FILE.exists():
        return {(r["learning_rate"], r["init_seed"]): r
                for r in json.loads(RESULTS_FILE.read_text(encoding="utf-8"))["candidates"]}
    return {}


def save_records(records, reference):
    """Write after every candidate, so an interrupted search resumes where it stopped."""
    TUNING.mkdir(parents=True, exist_ok=True)
    ordered = [records[(lr, seed)] for lr in LEARNING_RATES for seed in INIT_SEEDS
               if (lr, seed) in records]
    RESULTS_FILE.write_bytes(json.dumps({
        "grid": {"learning_rates": LEARNING_RATES, "init_seeds": INIT_SEEDS},
        "task": "extended task, correct labels, 1,000,000 sequences, 31,250 updates",
        "selection_rule": SELECTION_RULE,
        "selection_objective": "query-label accuracy only",
        "varies_with_init_seed": ("the backbone weights only; the two-way symbol head keeps "
                                  "its own fixed seed (11) and the training data is identical "
                                  "across candidates"),
        "scored_on": str(common.DEV_EVALUATOR_FILE.relative_to(common.ROOT)),
        "fixed_settings": {name: reference.get(name) for name in SHARED_SETTINGS},
        "candidates": ordered,
    }, indent=2, default=str).encode("utf-8"))


def run_grid(opts, dry_run):
    """Train and score every grid cell that does not already have a result."""
    features = common.load_features()
    splits = main_utils.get_splits_from_opts(opts, features.shape)
    reference = dict(vars(opts))
    reference.update(task="extended (query label, next symbol, its label)",
                     symbol_head_init_seed=pipeline.SYMBOL_HEAD_INIT_SEED,
                     training_choice_seed=pipeline.TRAINING_CHOICE_SEED,
                     dev_choice_seed=pipeline.DEV_CHOICE_SEED)
    expected_iterations = opts.train_iters
    TUNING.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if not dry_run:
        save_records(records, reference)          # the rule is on disk before any new result

    dataset, dev = None, None
    for learning_rate in LEARNING_RATES:
        for init_seed in INIT_SEEDS:
            previous = records.get((learning_rate, init_seed))
            if previous is not None and previous["status"] == "completed":
                continue
            folder = TUNING / candidate_name(learning_rate, init_seed)
            reused = find_reusable_run(learning_rate, init_seed, reference, expected_iterations)

            if dry_run:
                print("{:<40} {}".format(candidate_name(learning_rate, init_seed),
                                         "reuse " + str(reused.relative_to(common.ROOT))
                                         if reused else "train"))
                continue

            if dataset is None:                   # built once, on the first candidate that trains
                dev = pipeline.load_dev_set(opts)
                dataset = load_shared_dataset(opts, splits, reference)

            record = {"learning_rate": learning_rate, "init_seed": init_seed}
            if reused is not None:
                record.update(run=str(reused.relative_to(common.ROOT)), reused=True)
            else:
                print("training", candidate_name(learning_rate, init_seed), flush=True)
                # A finished folder would already have been reused, so anything here
                # is a leftover from an interrupted run. Refuse rather than resume.
                if folder.exists():
                    raise SystemExit("{} exists but has no final checkpoint. Move or delete "
                                     "it before re-running.".format(folder))
                record.update(run=str(folder.relative_to(common.ROOT)), reused=False)
                try:
                    train_candidate(folder, opts, learning_rate, init_seed, dataset,
                                    features, dev)
                except AssertionError as failure:
                    # train_generation asserts the loss stayed finite, so this is
                    # the learning rate diverging, not the code breaking.
                    record.update(status="failed", failure_kind="numerical",
                                  failure=str(failure))
                    records[(learning_rate, init_seed)] = record
                    save_records(records, reference)
                    continue
                if not is_complete(folder, expected_iterations):
                    record.update(status="failed", failure_kind="implementation",
                                  failure="training finished without a final checkpoint")
                    records[(learning_rate, init_seed)] = record
                    save_records(records, reference)
                    continue

            matches, why = configuration_matches(record["run"], learning_rate, init_seed,
                                                 reference)
            if not matches:
                record.update(status="failed", failure_kind="implementation",
                              failure="configuration mismatch: " + why)
                records[(learning_rate, init_seed)] = record
                save_records(records, reference)
                continue

            scores = score_final_checkpoint(common.ROOT / record["run"], opts, learning_rate,
                                            init_seed, dev)
            record["status"] = "completed"
            record["dev_query_label_accuracy"] = scores["query_label_accuracy"]
            record["dev_query_label_loss"] = scores["query_label_loss"]
            record["dev_symbol_loss"] = scores["symbol_loss"]
            record["dev_next_label_accuracy"] = scores["next_label_accuracy"]
            record["dev_next_label_loss"] = scores["next_label_loss"]
            records[(learning_rate, init_seed)] = record
            save_records(records, reference)
            print("  query acc {:.4f}  query loss {:.4f}  symbol loss {:.4f}  "
                  "next acc {:.4f}".format(scores["query_label_accuracy"],
                                           scores["query_label_loss"],
                                           scores["symbol_loss"],
                                           scores["next_label_accuracy"]), flush=True)

    return None if dry_run else records


def write_selection(records, summary, winner):
    """Record the winning rate and the exact seed-5 run the recursive chain needs."""
    chosen = records[(winner["learning_rate"], GENERATION_0_SEED)]
    SELECTION_FILE.write_bytes(json.dumps({
        "selection_rule": SELECTION_RULE,
        "selection_objective": "query-label accuracy only",
        "scored_on": str(common.DEV_EVALUATOR_FILE.relative_to(common.ROOT)),
        "selected_learning_rate": winner["learning_rate"],
        "claim": "best among the tested learning rates under this training budget",
        "winner_at_grid_boundary": winner["learning_rate"] in (min(LEARNING_RATES),
                                                               max(LEARNING_RATES)),
        "mean_query_accuracy": winner["mean_query_accuracy"],
        "per_seed_query_accuracy": winner["per_seed_query_accuracy"],
        "generation_0_seed": GENERATION_0_SEED,
        "generation_0_run": chosen["run"],
        "per_rate_summary": summary,
    }, indent=2).encode("utf-8"))


def print_summary(summary, winner):
    """One compact table: the selection objective plus the recorded side measures."""
    print("\nmean final scores on 1,000 development questions, over seeds 5, 6, 7")
    print("  {:<8} {:>9} {:>8} {:>10} {:>11} {:>9}".format(
        "rate", "query acc", "std", "query loss", "symbol loss", "next acc"))
    for row in summary:
        if row["mean_query_accuracy"] is None:
            print("  {:<8} {:>9}".format("{:g}".format(row["learning_rate"]), "no data"))
            continue
        print("  {:<8} {:>9.4f} {:>8.4f} {:>10.4f} {:>11.4f} {:>9.4f}".format(
            "{:g}".format(row["learning_rate"]), row["mean_query_accuracy"],
            row["std_query_accuracy"], row["mean_query_loss"],
            row["mean_symbol_loss"], row["mean_next_label_accuracy"]))
    print("  selected: {:g}".format(winner["learning_rate"]) if winner else "  no selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be trained, reused or skipped, then stop")
    args = parser.parse_args()
    common.use_above_normal_priority()

    if not common.DEV_EVALUATOR_FILE.exists():
        raise SystemExit("Missing evaluator. Run scripts/prepare_evaluation_data.py first.")

    opts = common.baseline_options()
    opts.model_output_classes = opts.fs_relabel

    # Train and score every outstanding grid cell, then apply the rule fixed above.
    records = run_grid(opts, args.dry_run)
    if records is None:
        return
    summary = summarise_by_learning_rate(records)
    winner = select_learning_rate(summary)
    if winner is None:
        raise SystemExit("No learning rate has all three seeds completed; not selecting.")
    write_selection(records, summary, winner)
    plot_summary(summary, winner, FIGURE_FILE)
    print_summary(summary, winner)
    print("\nwritten:", SELECTION_FILE.relative_to(common.ROOT))
    print("written:", FIGURE_FILE.relative_to(common.ROOT))


if __name__ == "__main__":
    main()
