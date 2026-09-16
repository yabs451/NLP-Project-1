"""Train one model on the original task by running the authors' main.py.

Uses the authors' published baseline arguments, changing only the output
location, which evaluators run, and (optionally) the learning rate, the
initialisation seed and the checkpoint schedule.

Usage (from the project root):
  .venv/Scripts/python.exe scripts/base_task/train_original.py --run-name my_run
  .venv/Scripts/python.exe scripts/base_task/train_original.py --run-name my_run --dry-run
"""
import argparse
from pathlib import Path
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, UPSTREAM, DEV_EVALUATOR_FILE, baseline_arguments

RESULTS = ROOT / "results" / "base_task"
TRAIN_SEQUENCES = 1000000
BATCH_SIZE = 32

# Checkpoint policy for main mechanistic runs: initialisation, four early
# snapshots while the circuit is still forming, then every 20,000 sequences.
# 55 checkpoints in total, saved directly during training.
EARLY_CHECKPOINTS = [1000, 2000, 5000, 10000]
CHECKPOINT_INTERVAL = 20000


def mechanistic_checkpoint_schedule():
    """The requested sequence counts for a main run.

    main.py checks the schedule at batch boundaries, so a request lands on the
    next multiple of the batch size (1,000 becomes 1,024). Every multiple of
    20,000 is already a multiple of 32, so those land exactly.
    """
    interval = list(range(0, TRAIN_SEQUENCES + 1, CHECKPOINT_INTERVAL))
    return sorted(set(EARLY_CHECKPOINTS + interval))


def evaluator_arguments(args):
    """Replace the authors' evaluator set with our development protocol.

    Drops their `fsl_test_class`, so the classes we reserve are never scored,
    and loads our fixed 1,000-question development evaluator from file. This
    changes evaluation only: main.py splits the training seeds before it reads
    any evaluator option, so the training stream is unaffected.
    """
    if not DEV_EVALUATOR_FILE.exists():
        raise SystemExit("Missing {}. Run scripts/prepare_evaluation_data.py first."
                         .format(DEV_EVALUATOR_FILE))
    original = {"--pe_names": ["fsl_train", "fsl_val_rl", "fsl_train_valex", "fsl_test_class"],
                "--pe_classes": ["train", "train", "train", "test"],
                "--pe_exemplars": ["train", "train", "val", "train"],
                "--pe_fs_relabel_scheme": ["train", "val", "train", "train"],
                "--pe_burstiness": ["1", "1", "1", "1"]}
    for flag, values in original.items():
        start = args.index(flag) + 1
        assert args[start:start + 4] == values, "upstream evaluator list changed"
        args[start:start + 4] = values[:3]      # keep all but fsl_test_class
    return args + ["--load_eval_data", str(DEV_EVALUATOR_FILE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True, help="Folder name for this run")
    parser.add_argument("--results-subfolder", default=None,
                        help="Place the run inside this subfolder of results/base_task, "
                             "e.g. 'tuning'. A single folder name, not a path.")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override the constant Adam learning rate (published: 1e-05)")
    parser.add_argument("--init-seed", type=int, default=None,
                        help="Override the weight-initialisation seed (published: 5)")
    parser.add_argument("--save-checkpoints", choices=["mechanistic", "endpoints"],
                        default="mechanistic",
                        help="mechanistic: 55 snapshots for circuit analysis. "
                             "endpoints: only the first and last, for runs where just the "
                             "final model is compared.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the underlying command and stop")
    options = parser.parse_args()

    # Both names must be plain folder names: this is what stops a run writing
    # outside results/ and clobbering something else.
    name = options.run_name
    if Path(name).name != name or name in (".", ".."):
        parser.error("run-name must be a single folder name")
    destination = RESULTS
    if options.results_subfolder is not None:
        subfolder = options.results_subfolder
        if Path(subfolder).name != subfolder or subfolder in (".", ".."):
            parser.error("results-subfolder must be a single folder name")
        destination = RESULTS / subfolder
    folder = destination / name

    args = evaluator_arguments(baseline_arguments())
    args += ["--base_folder", str(destination), "--run", name]
    # Appended last so they override the published values; argparse keeps the
    # last occurrence of a repeated option.
    if options.learning_rate is not None:
        args += ["--lr", repr(options.learning_rate)]
    if options.init_seed is not None:
        args += ["--init_seed", str(options.init_seed)]
    if options.save_checkpoints == "mechanistic":
        args += ["--ckpt_sched"] + [str(i) for i in mechanistic_checkpoint_schedule()]
    else:
        # A schedule of [0] plus the final checkpoint main.py always writes.
        args += ["--ckpt_every", str(TRAIN_SEQUENCES)]

    command = [sys.executable, "-u", str(UPSTREAM / "main.py"), *args]
    print(subprocess.list2cmdline(command), flush=True)
    if options.dry_run:
        return

    # exist_ok=False is deliberate: never write into an existing run folder.
    folder.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(JAX_PLATFORMS="cpu", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    # Output streams straight to the terminal; main.py writes config.json,
    # log.h5 and the checkpoints itself.
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=environment).returncode)


if __name__ == "__main__":
    main()
