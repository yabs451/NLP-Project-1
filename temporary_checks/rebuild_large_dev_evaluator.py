"""Rebuild the retired 10,000-question development evaluator.

Retired from the main workflow: the project now uses the 1,000-question set for
everything. Kept so the one-off evaluator-size comparison can be repeated.

The .h5 itself is not stored — it is ~60 MB of regenerable data. Running this
script recreates it byte-for-byte from the recorded seed and the unchanged
development classes.

Nothing under scripts/ imports this file.

Usage (from the project root):
  .venv/Scripts/python.exe temporary_checks/rebuild_large_dev_evaluator.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import common
from prepare_evaluation_data import build_evaluator, check_evaluator, select_classes

import h5py
import numpy as np

# The original recipe, copied from the protocol record before the evaluator was
# retired: same 100 development classes and task rules as the 1,000-question
# set, differing only in size and seed.
EVALUATOR_NAME = "fsl_dev_class_large"
SEQUENCES = 10000
SEED = 3007
OUTPUT = Path(__file__).resolve().parent / "eval_dev_large.h5"


def main():
    opts = common.baseline_options()
    features = common.load_features()
    selection = select_classes(opts, features.shape[0])
    splits = selection["splits"]

    evaluator = build_evaluator(opts, splits, selection["dev"], features, SEED, SEQUENCES)
    checks = check_evaluator(evaluator, np.asarray(features), selection["dev"], opts,
                             splits["relabeling"]["train"], SEQUENCES)
    with h5py.File(OUTPUT, "w") as handle:
        for field in ("examples", "labels"):
            handle.create_dataset("/".join([EVALUATOR_NAME, field]),
                                  data=np.asarray(evaluator[field]))
    print("wrote", OUTPUT.relative_to(ROOT), "with", checks["sequences"], "questions")


if __name__ == "__main__":
    main()
