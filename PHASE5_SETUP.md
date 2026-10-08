# Phase 5 final-test setup

## Files changed

- `scripts/evaluate_final_test.py` — new frozen final-checkpoint evaluator.
- `scripts/prepare_evaluation_data.py` — preserves `final_test_evaluator.built: true`
  when the final-test file already exists and preparation is later run without
  `--build-final-test`.
- `PHASE5_SETUP.md` — this setup record.

## What the evaluator does

`scripts/evaluate_final_test.py` has a fixed inventory containing 30 unique
models: five base generations, shared extended generation 0 once, and the 24
successors in the five frozen extended chains. It requires each run's last saved
checkpoint to equal its configured `train_iters`, then applies the existing
development scoring definitions to the reserved evaluator:

- base: query accuracy and query cross-entropy loss;
- extended: teacher-forced query-label accuracy/loss, following-label
  accuracy/loss, symbol loss, symbol-target accuracy, and symbol-position
  preference.

It does not train, generate data, calculate attention, or run ablations. It writes
one machine-readable file with the evaluator identity, frozen run/condition,
generation, checkpoint path and iteration, and metrics for every unique model.

## Commands for the frozen evaluation

From the repository root, build the reserved evaluator exactly once:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py --build-final-test
```

Then run the frozen evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_final_test.py
```

Expected files:

- preparation writes `results/evaluation_data/eval_final_test.h5`, rewrites the
  deterministic `eval_dev.h5`, and updates `class_splits.json`;
- evaluation writes `results/final_test/final_test_results.json`.

## Setup checks performed

- Parsed `scripts/evaluate_final_test.py` and
  `scripts/prepare_evaluation_data.py` with `ast.parse`: passed.
- Ran `scripts/evaluate_final_test.py --help`: exited 0.
- Ran `scripts/evaluate_final_test.py` while the reserved file was absent: it
  exited before checkpoint loading with the expected missing-file error.
- Imported the evaluator and ran its read-only frozen-run preflight: all 30 unique
  run folders exist, all resolve to checkpoint 1,000,000, and every checkpoint
  equals its saved `train_iters`.
- Confirmed that neither `results/evaluation_data/eval_final_test.h5` nor
  `results/final_test/final_test_results.json` exists after these checks.
- Ran `git diff --check` after the edits.

Reserved final test generated during setup: **NO**.

Reserved final test scored during setup: **NO**.

Models trained, datasets regenerated, experiments added, metrics changed, or
frozen conditions changed: **NO**.
