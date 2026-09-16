"""Learning-rate search for the base (label-only) task, and its evaluator comparison.

Trains a 5 x 3 grid (five learning rates, three initialisation seeds) on the
original task, then scores each final checkpoint. `--compare-evaluators` re-scores
the same saved models on the smaller development set and writes the comparison.

Method and results: findings/03_learning_rate_search.md

Usage (from the project root):
  .venv/Scripts/python.exe scripts/base_task/tune_learning_rate.py
  .venv/Scripts/python.exe scripts/base_task/tune_learning_rate.py --dry-run
  .venv/Scripts/python.exe scripts/base_task/tune_learning_rate.py --compare-evaluators
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

import numpy as np

LEARNING_RATES = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001]
INIT_SEEDS = [5, 6, 7]
PUBLISHED_LEARNING_RATE = 0.00001
# Seed 5 is the authors' initialisation seed, fixed in advance as the one whose
# checkpoint becomes generation 0. It is NOT whichever seed scores best.
GENERATION_0_SEED = 5

TUNING = common.ROOT / "results" / "base_task" / "tuning"
RESULTS_FILE = TUNING / "results.json"
SELECTION_FILE = TUNING / "selection.json"
COMPARISON_FILE = TUNING / "evaluator_comparison.json"
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


def plot_comparison(summaries, winners, path):
    """Final accuracy against learning rate, one line per evaluator.

    `summaries` maps an evaluator label to its per-rate summary rows. Individual
    seeds are drawn as points so their spread can be compared with the gap
    between learning rates.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"10,000 questions": "tab:blue", "1,000 questions": "tab:orange"}
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for label, summary in summaries.items():
        colour = colours.get(label, "tab:green")
        for row in summary:
            for accuracy in row["per_seed_accuracy"].values():
                axis.scatter(row["learning_rate"], accuracy, s=22, color=colour,
                             alpha=0.55, zorder=3)
        rates = [row["learning_rate"] for row in summary if row["mean_accuracy"] is not None]
        means = [row["mean_accuracy"] for row in summary if row["mean_accuracy"] is not None]
        axis.plot(rates, means, color=colour, marker="o", markersize=5,
                  label="{}, mean of seeds 5/6/7".format(label), zorder=2)
    for label, winner in winners.items():
        if winner is not None:
            axis.axvline(winner["learning_rate"], color=colours.get(label, "tab:green"),
                         ls="--", alpha=0.5)
    axis.axhline(0.2, color="grey", ls=":", lw=1, label="chance over 5 labels (20%)")
    axis.axhline(0.5, color="grey", ls="--", lw=1, label="chance within context (50%)")
    axis.set_xscale("log")
    axis.set_xlabel("constant Adam learning rate (log scale)")
    axis.set_ylabel("final development accuracy")
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
        "scored_on": str(common.LARGE_DEV_EVALUATOR_FILE.relative_to(common.ROOT)),
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
                                            common.LARGE_DEV_EVALUATOR_FILE)
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
        "scored_on": str(common.LARGE_DEV_EVALUATOR_FILE.relative_to(common.ROOT)),
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


def compare_evaluators():
    """Re-score all 15 saved models on the smaller evaluator and compare.

    The 1,000-question scores are computed here by loading each saved final
    checkpoint; they are never read out of a training log or an earlier
    evaluation record. The 10,000-question scores are read back from the grid
    record and are not recomputed.
    """
    reference = vars(common.baseline_options())
    expected_iterations = reference["train_iters"]
    records = load_records()
    if len(records) != len(LEARNING_RATES) * len(INIT_SEEDS):
        raise SystemExit("Expected 15 grid records, found {}.".format(len(records)))

    fresh, recorded, candidates = {}, {}, []
    for learning_rate in LEARNING_RATES:
        for init_seed in INIT_SEEDS:
            record = records[(learning_rate, init_seed)]
            if record["status"] != "completed":
                raise SystemExit("Candidate {} did not complete.".format(
                    candidate_name(learning_rate, init_seed)))
            scored = score_final_checkpoint(common.ROOT / record["run"], expected_iterations,
                                            common.DEV_EVALUATOR_FILE)
            fresh[(learning_rate, init_seed)] = scored
            recorded[(learning_rate, init_seed)] = {"accuracy": record["dev_accuracy"],
                                                    "loss": record["dev_loss"]}
            candidates.append({
                "candidate": candidate_name(learning_rate, init_seed),
                "learning_rate": learning_rate,
                "init_seed": init_seed,
                "run": record["run"],
                "checkpoint_sequences": expected_iterations,
                "fresh_1000_question_accuracy": scored["accuracy"],
                "fresh_1000_question_loss": scored["loss"],
                "recorded_10000_question_accuracy": record["dev_accuracy"],
                "recorded_10000_question_loss": record["dev_loss"],
            })
            print("{:<38} 1,000q {:.4f}   10,000q {:.4f}".format(
                candidates[-1]["candidate"], scored["accuracy"], record["dev_accuracy"]),
                flush=True)

    summaries = {"1,000 questions": summarise_by_learning_rate(fresh),
                 "10,000 questions": summarise_by_learning_rate(recorded)}
    winners = {label: select_learning_rate(summary) for label, summary in summaries.items()}

    COMPARISON_FILE.write_bytes(json.dumps({
        "purpose": ("Compare the two development evaluators on the same 15 saved models. The "
                    "1,000-question scores were computed fresh from the saved final "
                    "checkpoints; the 10,000-question scores are read from results.json."),
        "evaluators": {
            "1,000 questions": {
                "file": str(common.DEV_EVALUATOR_FILE.relative_to(common.ROOT)),
                "source": "freshly evaluated from saved final checkpoints"},
            "10,000 questions": {
                "file": str(common.LARGE_DEV_EVALUATOR_FILE.relative_to(common.ROOT)),
                "source": "read from results.json, not recomputed"}},
        "selection_rule": ("highest mean accuracy over seeds 5, 6 and 7; exact ties use lowest "
                           "mean loss; remaining ties prefer 1e-05, otherwise the smaller rate"),
        "generation_0_seed": GENERATION_0_SEED,
        "selected_learning_rate": {label: (None if w is None else w["learning_rate"])
                                   for label, w in winners.items()},
        "selection_agrees_across_evaluators": len({
            None if w is None else w["learning_rate"] for w in winners.values()}) == 1,
        "per_rate_summary": summaries,
        "candidates": candidates,
    }, indent=2).encode("utf-8"))

    plot_comparison(summaries, winners, FIGURE_FILE)
    return summaries, winners


def print_summary(summaries, winners):
    """One compact table per evaluator, plus the winner under each."""
    for label, summary in summaries.items():
        print("\n{} -- mean accuracy over seeds 5, 6, 7".format(label))
        print("  {:<10} {:>8} {:>8} {:>10}  per-seed".format("rate", "mean", "std", "mean loss"))
        for row in summary:
            print("  {:<10} {:>8.4f} {:>8.4f} {:>10.4f}  {}".format(
                "{:g}".format(row["learning_rate"]), row["mean_accuracy"],
                row["std_accuracy"], row["mean_loss"],
                {s: round(a, 4) for s, a in row["per_seed_accuracy"].items()}))
        winner = winners[label]
        print("  selected: {:g}".format(winner["learning_rate"]) if winner else "  no selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be trained, reused or skipped, then stop")
    parser.add_argument("--compare-evaluators", action="store_true",
                        help="Re-score the 15 saved models on the 1,000-question set and "
                             "compare with the recorded 10,000-question results")
    args = parser.parse_args()

    if not common.LARGE_DEV_EVALUATOR_FILE.exists():
        raise SystemExit("Missing evaluators. Run scripts/prepare_evaluation_data.py first.")

    if args.compare_evaluators:
        summaries, winners = compare_evaluators()
        print_summary(summaries, winners)
        print("\nwritten:", COMPARISON_FILE.relative_to(common.ROOT))
        print("written:", FIGURE_FILE.relative_to(common.ROOT))
        return

    # Train and score every outstanding grid cell, then apply the selection rule
    # to the 10,000-question scores and record the chosen checkpoint.
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
    print_summary({"10,000 questions": summary}, {"10,000 questions": winner})
    print("\nwritten:", SELECTION_FILE.relative_to(common.ROOT))


if __name__ == "__main__":
    main()
