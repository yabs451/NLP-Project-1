"""Score a finished run's checkpoint on the fixed 1,000-question development set.

Accuracy is argmax over all five labels (chance 20%); `in_context_acc` restricts
the argmax to the two labels present in the context (chance 50%); loss is mean
query cross-entropy in nats.

Usage (from the project root):
  .venv/Scripts/python.exe scripts/evaluate_on_dev.py results/base_task/<run>
  .venv/Scripts/python.exe scripts/evaluate_on_dev.py results/base_task/<run> --checkpoint 500000
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Sequence count of the checkpoint to score (default: the last one)")
    args = parser.parse_args()

    folder = args.run_folder.resolve()
    evaluator = common.DEV_EVALUATOR_FILE
    if not evaluator.exists():
        raise SystemExit("Missing {}. Run scripts/prepare_evaluation_data.py first."
                         .format(evaluator))

    # Default to the final checkpoint: that is what every comparison uses, since
    # the project deliberately does no best-checkpoint selection.
    iteration = args.checkpoint
    if iteration is None:
        iteration = common.available_checkpoints(folder)[-1]

    scored = common.score_checkpoint(folder, iteration, evaluator)
    result = {"run": str(folder.relative_to(common.ROOT)),
              "checkpoint_sequences": iteration,
              "evaluator_file": str(evaluator.relative_to(common.ROOT)),
              "metrics": scored}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
