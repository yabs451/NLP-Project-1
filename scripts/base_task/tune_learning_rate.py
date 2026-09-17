"""Learning-rate search for the base (label-only) task.

Trains a 6 x 3 grid (six learning rates, three initialisation seeds) on the
original task and scores each final checkpoint on the fixed 1,000-question
development set.

Method and results: findings/01_learning_rate_search.md

Usage (from the project root):
  .venv/Scripts/python.exe scripts/base_task/tune_learning_rate.py
  .venv/Scripts/python.exe scripts/base_task/tune_learning_rate.py --dry-run
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

import numpy as np

# 1e-3 was added after the first five rates put the winner at the top of the
# range. The search stops here: no further rate is tested.
LEARNING_RATES = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.001]
INIT_SEEDS = [5, 6, 7]
PUBLISHED_LEARNING_RATE = 0.00001
# Seed 5 is the authors' initialisation seed, fixed in advance as the one whose
# checkpoint becomes generation 0. It is NOT whichever seed scores best.
GENERATION_0_SEED = 5

TUNING = common.ROOT / "results" / "base_task" / "tuning"
RESULTS_FILE = TUNING / "results.json"
SELECTION_FILE = TUNING / "selection.json"
FIGURE_FILE = TUNING / "learning_rate_comparison.png"
TRAINER = Path(__file__).resolve().parent / "train_original.py"

# Settings every candidate must share for its result to be comparable, checked
# against a run's own config.json before that run is reused or accepted.
SHARED_SETTINGS = ["train_seed", "eval_seed", "train_iters", "train_bs", "optimizer",
                   "weight_decay", "d_model", "num_heads", "depth", "mlp_ratio",
                   "class_split", "exemplar_split", "fs_relabel", "fs_relabel_split_seed",
                   "train_context_len", "noise_scale_train"]


def candidate_name(learning_rate, init_seed):
    """Folder name stating the two settings that vary. '1e-06' means 0.000001."""
    return "learning_rate_{:g}_init_seed_{}".format(learning_rate, init_seed)


def configuration_matches(run_folder, learning_rate, init_seed, reference):
    """Check a finished run really has the configuration this candidate needs.

    Returns (matches, reason). Runs trained on generated data are rejected: a
    successor's config.json looks the same, but its targets came from a parent
    model rather than from the task.
    """
    run_folder = Path(run_folder)
    if (run_folder / "generation_metadata.json").exists():
        return False, "trained on generated targets, not the original task"
    if not (run_folder / "config.json").exists():
        return False, "no config.json"
    config = json.loads((run_folder / "config.json").read_text())
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
    """Find an already-finished run with this configuration, if one exists.

    Searches the other base-task runs as well as the tuning folder, so the
    published baseline can serve as its own grid cell instead of being retrained.
    """
    base = common.ROOT / "results" / "base_task"
    for folder in sorted(base.glob("*")) + sorted(TUNING.glob("*")):
        if not folder.is_dir() or folder == TUNING:
            continue
        matches, _ = configuration_matches(folder, learning_rate, init_seed, reference)
        if matches and is_complete(folder, expected_iterations):
            return folder
    return None


def train_candidate(folder, learning_rate, init_seed):
    """Train one candidate by calling our ordinary trainer as a subprocess."""
    command = [sys.executable, "-u", str(TRAINER),
               "--results-subfolder", TUNING.name, "--run-name", folder.name,
               "--learning-rate", repr(learning_rate), "--init-seed", str(init_seed),
               # Tuning compares final checkpoints only, so the mechanistic
               # schedule is not used here.
               "--save-checkpoints", "endpoints"]
    return subprocess.run(command, cwd=common.ROOT).returncode


def score_final_checkpoint(run_folder, expected_iterations, evaluator_file):
    """Load a candidate's final checkpoint and score it on a saved evaluator."""
    scored = common.score_checkpoint(run_folder, expected_iterations, evaluator_file)
    name = list(scored)[0]
    return {"accuracy": scored[name]["acc"], "loss": scored[name]["loss"]}


def summarise_by_learning_rate(scores):
    """Average each learning rate over its seeds.

    `scores` maps (learning_rate, init_seed) -> {"accuracy", "loss"}. Returns one
    row per learning rate, in grid order.
    """
    summary = []
    for learning_rate in LEARNING_RATES:
        done = {seed: scores[(learning_rate, seed)] for seed in INIT_SEEDS
                if (learning_rate, seed) in scores}
        accuracies = [row["accuracy"] for row in done.values()]
        losses = [row["loss"] for row in done.values()]
        summary.append({
            "learning_rate": learning_rate,
            "seeds_scored": sorted(done),
            "per_seed_accuracy": {seed: done[seed]["accuracy"] for seed in sorted(done)},
            "per_seed_loss": {seed: done[seed]["loss"] for seed in sorted(done)},
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else None,
            "std_accuracy": float(np.std(accuracies)) if accuracies else None,
            "mean_loss": float(np.mean(losses)) if losses else None,
        })
    return summary


def select_learning_rate(summary):
    """Apply the selection rule: highest mean accuracy, then lowest mean loss.

    Remaining ties prefer the authors' 1e-05, otherwise the smaller rate. Only
    learning rates with all three seeds are eligible, because averaging over a
    subset would flatter a rate whose other seeds are missing.
    """
    eligible = [row for row in summary if len(row["seeds_scored"]) == len(INIT_SEEDS)]
    if not eligible:
        return None
    best = max(row["mean_accuracy"] for row in eligible)
    tied = [row for row in eligible if row["mean_accuracy"] == best]
    if len(tied) > 1:
        lowest = min(row["mean_loss"] for row in tied)
        tied = [row for row in tied if row["mean_loss"] == lowest]
    if len(tied) > 1:
        published = [row for row in tied if row["learning_rate"] == PUBLISHED_LEARNING_RATE]
        tied = published or [min(tied, key=lambda row: row["learning_rate"])]
    return tied[0]


def plot_summary(summary, winner, path):
    """Final accuracy against learning rate, from the 1,000-question evaluator.

    Individual seeds are drawn as points so their spread can be compared with
    the gap between learning rates.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for row in summary:
        for accuracy in row["per_seed_accuracy"].values():
            axis.scatter(row["learning_rate"], accuracy, s=24, color="tab:blue",
                         alpha=0.55, zorder=3)
    rates = [row["learning_rate"] for row in summary if row["mean_accuracy"] is not None]
    means = [row["mean_accuracy"] for row in summary if row["mean_accuracy"] is not None]
    axis.plot(rates, means, color="tab:blue", marker="o", markersize=5,
              label="mean of seeds 5, 6, 7", zorder=2)
    if winner is not None:
        axis.axvline(winner["learning_rate"], color="tab:red", ls="--", alpha=0.6,
                     label="selected: {:g}".format(winner["learning_rate"]))
    axis.axhline(0.2, color="grey", ls=":", lw=1, label="chance over 5 labels (20%)")
    axis.axhline(0.5, color="grey", ls="--", lw=1, label="chance within context (50%)")
    axis.set_xscale("log")
    axis.set_xlabel("constant Adam learning rate (log scale)")
    axis.set_ylabel("final accuracy on 1,000 development questions")
    axis.set_title("Learning-rate search: final accuracy after 31,250 updates")
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def load_records():
    """Previous grid results, keyed by (learning rate, seed) so retries replace them."""
    if RESULTS_FILE.exists():
        return {(r["learning_rate"], r["init_seed"]): r
                for r in json.loads(RESULTS_FILE.read_text())["candidates"]}
    return {}


def save_records(records, reference):
    """Write after every candidate, so an interrupted search can resume."""
    TUNING.mkdir(parents=True, exist_ok=True)
    ordered = [records[(lr, seed)] for lr in LEARNING_RATES for seed in INIT_SEEDS
               if (lr, seed) in records]
    RESULTS_FILE.write_bytes(json.dumps({
        "grid": {"learning_rates": LEARNING_RATES, "init_seeds": INIT_SEEDS},
        "training": "original task, true query targets, 1,000,000 sequences, 31,250 updates",
        "scored_on": str(common.DEV_EVALUATOR_FILE.relative_to(common.ROOT)),
        "fixed_settings": {name: reference.get(name) for name in SHARED_SETTINGS},
        "candidates": ordered,
    }, indent=2).encode("utf-8"))


def run_grid(dry_run):
    """Train and score every grid cell that does not already have a result."""
    reference = vars(common.baseline_options())
    expected_iterations = reference["train_iters"]
    TUNING.mkdir(parents=True, exist_ok=True)
    records = load_records()

    for learning_rate in LEARNING_RATES:
        for init_seed in INIT_SEEDS:
            previous = records.get((learning_rate, init_seed))
            if previous is not None and previous["status"] == "completed":
                continue
            folder = TUNING / candidate_name(learning_rate, init_seed)
            reused = find_reusable_run(learning_rate, init_seed, reference, expected_iterations)

            if dry_run:
                action = "reuse " + str(reused.relative_to(common.ROOT)) if reused else "train"
                print("{:<40} {}".format(candidate_name(learning_rate, init_seed), action))
                continue

            record = {"learning_rate": learning_rate, "init_seed": init_seed}
            if reused is not None:
                record.update(run=str(reused.relative_to(common.ROOT)), reused=True,
                              status="completed")
            else:
                print("training", candidate_name(learning_rate, init_seed), flush=True)
                # A finished folder would already have been reused, so anything
                # here is a leftover from an interrupted run. Refuse rather than
                # resume into it or treat it as a result.
                if folder.exists():
                    raise SystemExit("{} exists but has no final checkpoint. Move or delete "
                                     "it before re-running.".format(folder))
                code = train_candidate(folder, learning_rate, init_seed)
                record.update(run=str(folder.relative_to(common.ROOT)), reused=False)
                if code != 0 or not is_complete(folder, expected_iterations):
                    # A crash is an implementation failure, not a bad learning
                    # rate, so record it rather than dropping the candidate.
                    record.update(status="failed", failure="trainer exited {}".format(code))
                    records[(learning_rate, init_seed)] = record
                    save_records(records, reference)
                    continue
                record["status"] = "completed"

            matches, why = configuration_matches(record["run"], learning_rate, init_seed, reference)
            if not matches:
                record.update(status="failed", failure="configuration mismatch: " + why)
                records[(learning_rate, init_seed)] = record
                save_records(records, reference)
                continue
            scored = score_final_checkpoint(common.ROOT / record["run"], expected_iterations,
                                            common.DEV_EVALUATOR_FILE)
            record["dev_accuracy"] = scored["accuracy"]
            record["dev_loss"] = scored["loss"]
            records[(learning_rate, init_seed)] = record
            save_records(records, reference)

    return None if dry_run else records


def write_selection(records, summary, winner):
    """Record the winning learning rate and the exact generation-0 checkpoint."""
    expected_iterations = vars(common.baseline_options())["train_iters"]
    chosen = records[(winner["learning_rate"], GENERATION_0_SEED)]
    checkpoint = common.checkpoint_path(common.ROOT / chosen["run"], expected_iterations)
    SELECTION_FILE.write_bytes(json.dumps({
        "selection_rule": ("highest mean final accuracy over init seeds 5, 6 and 7, compared "
                           "unrounded; ties broken by lowest mean loss, then by preferring "
                           "1e-05, then the smaller rate"),
        "scored_on": str(common.DEV_EVALUATOR_FILE.relative_to(common.ROOT)),
        "selected_learning_rate": winner["learning_rate"],
        "claim": "best among the tested learning rates under this training budget",
        "winner_at_grid_boundary": winner["learning_rate"] in (min(LEARNING_RATES),
                                                               max(LEARNING_RATES)),
        "mean_accuracy": winner["mean_accuracy"],
        "per_seed_accuracy": winner["per_seed_accuracy"],
        "generation_0_seed": GENERATION_0_SEED,
        "generation_0_run": chosen["run"],
        "generation_0_checkpoint": str(checkpoint.relative_to(common.ROOT)),
        "per_rate_summary": summary,
    }, indent=2).encode("utf-8"))


def print_summary(summary, winner):
    """One compact table: mean, spread and per-seed accuracy for each rate."""
    print("\nmean accuracy on 1,000 development questions, over seeds 5, 6, 7")
    print("  {:<10} {:>8} {:>8} {:>10}  per-seed".format("rate", "mean", "std", "mean loss"))
    for row in summary:
        if row["mean_accuracy"] is None:
            print("  {:<10} {:>8}".format("{:g}".format(row["learning_rate"]), "no data"))
            continue
        print("  {:<10} {:>8.4f} {:>8.4f} {:>10.4f}  {}".format(
            "{:g}".format(row["learning_rate"]), row["mean_accuracy"],
            row["std_accuracy"], row["mean_loss"],
            {seed: round(a, 4) for seed, a in row["per_seed_accuracy"].items()}))
    print("  selected: {:g}".format(winner["learning_rate"]) if winner else "  no selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be trained, reused or skipped, then stop")
    args = parser.parse_args()

    if not common.DEV_EVALUATOR_FILE.exists():
        raise SystemExit("Missing evaluator. Run scripts/prepare_evaluation_data.py first.")

    # Train and score every outstanding grid cell, then apply the selection rule
    # and record the chosen checkpoint.
    records = run_grid(args.dry_run)
    if records is None:
        return
    summary = summarise_by_learning_rate(
        {key: {"accuracy": r["dev_accuracy"], "loss": r["dev_loss"]}
         for key, r in records.items() if r["status"] == "completed"})
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
