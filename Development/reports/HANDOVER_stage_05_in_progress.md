# Stage 05 handover — IN PROGRESS, not finished

Written 16 September 2026. Delete this file once stage 05 is complete.

## Read first

`CLAUDE.md`, `README.md`, `reports/04_project_cleanup_and_first_successor.md`.

## What stage 05 is

Build a second, larger development evaluator and run a learning-rate search, then
select the winning setting and identify the matching generation-0 checkpoint.
**Do not train recursive successors and do not implement the extended task** —
six successors are a later task.

## Done already

- **10,000-question development evaluator** built and recorded.
  `results/evaluation_data/eval_dev_large.h5`, evaluator name `fsl_dev_class_large`,
  seed **3007**, recorded in `class_splits.json` before any candidate was inspected.
  Same 100 development classes and task rules as the existing 1,000-question set.
  The 1,000-question set is unchanged (verified: generation 0 still re-scores to
  its logged metrics).
- **New/changed code**
  - `scripts/base_task/tune_learning_rate.py` (new) — the 5x3 grid driver,
    selection rule, and comparison figure.
  - `scripts/evaluate_on_dev.py` (new) — score any checkpoint on either dev set.
  - `scripts/common.py` — added `LARGE_DEV_EVALUATOR_FILE`, `load_evaluator_file()`,
    `score_checkpoint()`.
  - `scripts/prepare_evaluation_data.py` — builds both dev evaluators.
  - `scripts/base_task/train_original.py` — added `--learning-rate`, `--init-seed`,
    `--save-checkpoints {all,endpoints}`, `--results-subfolder`.
  - `CLAUDE.md` — testing/scratch conventions section.
  - `README.md` — two-evaluator explanation, tuning commands, new files.
- **Corrections applied** to `findings/02` and `reports/04`: "identical" softened to
  "close but not identical" with exact differences; timings marked approximate;
  Git section updated (stage 04 *is* committed and pushed as `ed6d2ba`).

## Still to do

1. Finish the grid (see below).
2. Re-run `tune_learning_rate.py` once more after completion — it skips finished
   candidates, applies the selection rule, writes `selection.json` and
   `learning_rate_comparison.png`.
3. Copy evidence into `findings/evidence/` (results table, selection record, figure).
4. Write `findings/03_learning_rate_search.md`.
5. Write `reports/05_development_evaluation_and_tuning.md`.
6. Update README with the selected rate and the generation-0 checkpoint path.

## Resuming the grid

```powershell
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py
```

It resumes automatically: results are saved after every candidate and finished
ones are skipped. **Nothing needs retraining.**

Progress at handover: **4 of 15 complete**.

| learning rate | seed 5 | seed 6 | seed 7 |
| --- | ---: | ---: | ---: |
| 1e-06 | 0.5224 | 0.5184 | 0.5184 |
| 3e-06 | 0.8383 | not run | not run |
| 1e-05 | not run | not run | not run |
| 3e-05 | not run | not run | not run |
| 0.0001 | not run | not run | not run |

The 1e-06 results are **genuine under-training**, not a failure: loss stayed
finite and was still falling, the model just cannot converge at that rate within
31,250 updates. All soundness checks passed on those runs.

## Selection rule (fixed before results were inspected)

Highest mean final development accuracy over seeds 5, 6 and 7 on the
10,000-question set, unrounded; ties broken by lowest mean development loss,
then by preferring the authors' 1e-05, then the smaller rate. A rate missing any
seed is excluded — never averaged over a subset. Verified against six synthetic
cases including all tie-break branches.

**Generation 0 is the winning rate's SEED-5 checkpoint** — seed 5 is fixed in
advance, not whichever seed scored best.

## Environment problem: OneDrive

The project sits inside OneDrive. `results/` holds ~1.8 GB (two runs of 1,001
checkpoints each), and OneDrive's sync service competes for CPU with training.
Two candidates doing identical work took **7.7 min and 83 min**.

Mitigation in place: `results/scratch/raise_training_priority.ps1` (ignored,
temporary, not a reproduction step) sets the training process to `AboveNormal`
every 20 s. That restored ~81,000 sequences/min, about 12 min per candidate.

Better fixes for the user: pause OneDrive while running, or move the project out
of OneDrive entirely. **All quoted runtimes in this stage are contended and are
not clean benchmarks.**

If the watcher is not running, restart it:

```powershell
Start-Process powershell -ArgumentList '-NoProfile','-WindowStyle','Hidden','-File','"<project>\results\scratch\raise_training_priority.ps1"' -WindowStyle Hidden
```

## Known issue for the six-successor stage

`run_recursive.py` names a successor `generation_1_generated_data_init_seed_5`
based only on the generation number and seed, ignoring which parent it came from
(`scripts/base_task/run_recursive.py:330`). If the tuning winner is not 1e-05,
the new generation 0 differs from the pilot, but its successor would want the
same folder name as the existing pilot successor. It fails safely — the script
refuses to overwrite — but the next stage must decide on a naming scheme, for
example putting tuned-lineage runs under `results/base_task/tuned/`.

## Failures recorded honestly

An earlier full-grid attempt failed on all 14 trainable candidates because the
driver passed `tuning/<name>` to `--run-name`, which correctly refuses path
separators. That was an implementation bug, not numerical divergence. It is
fixed (`--results-subfolder`), and the failed record is preserved at
`results/scratch/tuning_results_first_attempt_failed.json`.

## Git

Nothing from stage 05 is committed. Last commit is `ed6d2ba` (stage 04), which
is on GitHub. Do not commit, push or reset unless asked.
