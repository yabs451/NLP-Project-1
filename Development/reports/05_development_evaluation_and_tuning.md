# 05 — Development evaluation, learning-rate tuning, and project cleanup

16 September 2026. Local operational report; not tracked in Git.

Scope: finish the outstanding tuning deliverables, re-evaluate all 15 saved
models on the 1,000-question development set, compare the two evaluators, and
simplify the project down to files with a current purpose. No model was
retrained, no successor was run, and the extended task was not implemented.

## File map

```
NLP-Project-1/
├── CLAUDE.md                          conventions (rewritten)
├── README.md                          reproduction guide (rewritten)
├── requirements.txt / -lock.txt       dependency pins and full freeze
├── scripts/
│   ├── common.py                      shared helpers
│   ├── prepare_evaluation_data.py     class splits + both development sets
│   ├── evaluate_on_dev.py             score one saved checkpoint
│   └── base_task/
│       ├── train_original.py          train one model on the original task
│       ├── tune_learning_rate.py      the grid, the selection, the comparison
│       ├── run_recursive.py           generate data from a parent, train successor
│       └── analyse_runs.py            curves, circuit measures, ablations
├── results/
│   ├── evaluation_data/               class_splits.json (tracked), both .h5 sets
│   └── base_task/
│       ├── generation_0_original_data_init_seed_5/
│       ├── generation_1_generated_data_init_seed_5/
│       └── tuning/                    14 candidate runs + 4 result files
├── findings/                          01, 02, 03
├── Development/                       local only, Git-ignored
│   ├── reports/                       01-04 + this report + old handover
│   └── outputs/                       inspection, protocol_checks, failed-attempt record
└── upstream/icl-dynamics/             the authors' code, unmodified
```

### Retained, and why

| Kept | Current purpose |
| --- | --- |
| `common.py`, `prepare_evaluation_data.py`, `evaluate_on_dev.py` | Imported by every script; build the evaluation data; score a saved model — all named in README reproduction steps |
| `train_original.py`, `tune_learning_rate.py`, `run_recursive.py`, `analyse_runs.py` | The four experiment/analysis entry points |
| `config.json`, `log.h5`, `checkpoints/` per run | Read by the analysis, the tuning and the recursive pipeline |
| `analysis/` per run | The figures and numbers findings 01 and 02 cite |
| `generation_metadata.json` | Holds target-quality statistics and worked examples that exist nowhere else; finding 02 cites them |
| `tuning/results.json`, `selection.json` | The grid record and the recorded 10,000-question selection |
| `evaluation_data/*.h5`, `class_splits.json` | The evaluation protocol and both question sets |
| `results/base_task/generation_1_...` | The pilot successor, cited by finding 02 |

### Moved to `Development/` (local, untracked, kept)

| Moved | Reason |
| --- | --- |
| `reports/01`–`04` → `Development/reports/` | Operational handovers, not scientific record |
| `HANDOVER_stage_05_in_progress.md` → `Development/reports/` | Superseded by this report; kept as history |
| `results/inspection/` → `Development/outputs/inspection/` | Setup-inspection dump; nothing reads it |
| `results/protocol_checks/` → `Development/outputs/protocol_checks/` | Stage-02 evidence for a claim already quoted in report 02; 52 MB of runs no current analysis reads |
| `results/scratch/tuning_results_first_attempt_failed.json` → `Development/outputs/` | Record of the stage-05 implementation failure |

### Removed, and why

| Removed | Dependency check | Reason |
| --- | --- | --- |
| `scripts/verify_run.py` | No importers; `verification.json` read by nothing | Standalone verification report, not part of the final project |
| `scripts/inspect_source_data.py` | No importers; `inspection.json` read by nothing | Setup-inspection utility; its facts are already quoted in reports 01/02 |
| 49 × `timing.json`, `console.log`, `command.json`, `verification.json` | `grep` across `scripts/` found writers only, no readers | Runtime files, debugging transcripts, argument duplication and verification reports |
| Timing collection in `train_original.py` and `run_recursive.py` | — | Runtime information is no longer collected at all, and was not replaced by a runtime column |
| `--full` / short-run mode, `--protocol` option, `--save_eval_data` | Only caller was `tune_learning_rate.py`, updated | Every real run used `--full --protocol assignment`; the alternatives were dead |
| `findings/evidence/` (8 files) | Byte-compared against the originals under `results/` first | Duplicated measurements and figures; findings now reference `results/` |
| `results/scratch/raise_training_priority.ps1`, `plot_smoke_test.png` | — | Temporary watcher and a smoke-test artefact |
| 937 + 937 checkpoints | See checkpoint section | Superseded by the new checkpoint policy |

Everything under `Development/` is ignored by Git, so the moves show in
`git status` as deletions of the old tracked paths, with local copies preserved.

## Upstream functions used

Unchanged, under `upstream/icl-dynamics/`: `main.py` `evaluate` (:200),
`eval_step` (:132), `make_batched_fn` (:183), `train_step` (:87), evaluator
construction (:356-400), and the `--ckpt_sched` option (`main_utils.py`:59) which
now carries our checkpoint policy. `main_utils.get_model_from_opts` (:222) and
`get_optimizer_from_opts` (:250) build the model and the constant-rate Adam.
Nothing in upstream was edited.

## Checkpoint policy

`train_original.py` now computes the schedule and passes it to `main.py` as
`--ckpt_sched`, so the 55 snapshots are written **during** training rather than
saved densely and pruned:

- initialisation (0)
- four early snapshots requested at 1,000 / 2,000 / 5,000 / 10,000, landing on
  batch boundaries at **1,024 / 2,016 / 5,024 / 10,016**
- every 20,000 through 1,000,000 (all already multiples of the batch size)

Verified: 55 requested values produce 55 distinct actual sequence counts.
`--save-checkpoints endpoints` remains for tuning, where only final models are
compared.

### Pruning generations 0 and 1

Both were trained before this policy with 1,001 checkpoints. Before deleting
anything, the checkpoints underlying the current figures were identified from
each run's `analysis.json` (`checkpoints_analysed`): 0, 1024, 2016, 3008, 5024,
9024, 14016, 25024, 42016, 71008, 120000, 204000, 347008, 589024, 1000000.

Kept per run = those 15 ∪ the 55 policy values (all present on disk) ∪ the final
checkpoint = **64 kept, 937 removed**.

**Figures remain reproducible.** Re-running `analyse_runs.py` on generation 0
after pruning selected the same 15 checkpoints and produced identical values
(dev accuracy 0.9670000672, L1H3 ablation 0.9280, induction deltas unchanged).

`results/` fell from 1.8 GB to 354 MB.

## Fresh evaluation of the 15 saved models

Command:

```powershell
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py --compare-evaluators
```

Each of the 15 final checkpoints was loaded and scored directly on
`results/evaluation_data/eval_dev.h5` through `common.score_checkpoint`, which
calls the authors' `evaluate`. No score was taken from a training log or an
earlier evaluation record, and the 10,000-question scores were read back from
`results.json` rather than recomputed or overwritten. No training occurred.

Mean accuracy over seeds 5, 6, 7:

| learning rate | 1,000 questions (fresh) | 10,000 questions (recorded) | mean loss 1,000q | mean loss 10,000q |
| --- | ---: | ---: | ---: | ---: |
| 1e-6 | 0.5157 | 0.5197 | 0.7593 | 0.7571 |
| 3e-6 | 0.8127 | 0.8161 | 0.4135 | 0.4189 |
| 1e-5 | 0.9677 | 0.9611 | 0.0928 | 0.1142 |
| 3e-5 | 0.9790 | 0.9766 | 0.0822 | 0.1037 |
| **1e-4** | **0.9923** | **0.9911** | **0.0278** | **0.0349** |

**Both evaluators select 1e-4**, with the same ranking and no crossings. The
per-candidate gap is within 1.1 accuracy points on 14 of 15 models; the outlier
is (1e-6, seed 6) at 2.3 points, in the band where models sit near chance.

**The winner and the interpretation did not change.** 1e-4 won the original
10,000-question search and lies at that grid's upper boundary; the fresh
1,000-question evaluation agrees. What the comparison adds is that the smaller
set would have been sufficient for a decision this clear-cut.

Seed 5 remains the predetermined parent seed. Selected generation-0 checkpoint:
`results/base_task/tuning/learning_rate_0.0001_init_seed_5/checkpoints/00001000000.eqx`.

## Outputs produced

- `results/base_task/tuning/evaluator_comparison.json` — per-candidate accuracy
  and loss on both evaluators, per-rate means and standard deviations, the
  winner under each, and whether they agree
- `results/base_task/tuning/learning_rate_comparison.png` — the previously
  missing figure, now showing both evaluators
- `findings/03_learning_rate_search.md` — the scientific write-up
- `README.md`, `CLAUDE.md` — rewritten

## Checks performed

| Check | Result |
| --- | --- |
| Active processes before editing | None training; leftover priority watcher stopped |
| Overrides reach the run | `--learning-rate` / `--init-seed` parsed by upstream's own parser and confirmed in the resolved options |
| Evaluator set after simplification | `fsl_train`, `fsl_val_rl`, `fsl_train_valex` plus the loaded dev set; `fsl_test_class` absent |
| Checkpoint schedule | 55 requested → 55 distinct actual counts |
| Figure reproducibility after pruning | Identical checkpoints selected and identical values |
| `findings/evidence` duplicates | Byte-identical to `results/` originals except one stale path field; deleted after confirming |
| Removed files had no readers | `grep` across `scripts/` before each removal |
| All scripts compile and run | Yes; `evaluate_on_dev.py` and the tuning dry-run exercised |
| Reserved final test | Not generated, not scored |

## Remaining dependencies and open decisions

1. **Which development evaluator to retain** — the user's decision, not made.
   Both are kept for now because the comparison was explicitly requested.
   Retiring the 10,000-question set would mean deleting `eval_dev_large.h5`, its
   build step and `LARGE_DEV_SEQUENCES` in `prepare_evaluation_data.py`, its seed
   record in `class_splits.json`, `LARGE_DEV_EVALUATOR_FILE` in `common.py`, and
   the `--compare-evaluators` path plus `results.json`'s `scored_on` field.
   Retiring the 1,000-question set is harder: it is loaded into every training
   run and appears in every `log.h5` learning curve, so retiring it would change
   what future runs monitor and break comparability with generations 0 and 1.
2. **Whether to retrain generation 0 at 1e-4** before the recursive chain.
   Generations 0 and 1 were trained at 1e-5 and are a pilot lineage.
3. **Whether `findings/` should be tracked** — untouched, as instructed.
4. **Successor naming collision**, still open from stage 04: `run_recursive.py`
   names a successor from the generation number and seed only
   (`run_recursive.py`, `run_one_generation`), ignoring which parent it came
   from. A successor of the tuned generation 0 would want the same folder name
   as the existing pilot. It fails safely — the script refuses to overwrite —
   but the six-successor stage needs a naming scheme decided first.

## Git status

Nothing was committed, pushed or reset; no remote was changed. Nothing is
staged, so the working tree can be reviewed as a whole. HEAD remains `ed6d2ba`.

Modified: `.gitignore`, `CLAUDE.md`, `README.md`, `findings/01`, `findings/02`,
`results/evaluation_data/class_splits.json`, `scripts/common.py`,
`scripts/prepare_evaluation_data.py`, `scripts/base_task/train_original.py`,
`scripts/base_task/run_recursive.py`.

Deleted from tracking (local copies kept under `Development/` where noted):
`reports/01`–`04`, `findings/evidence/*` (8 files), `scripts/verify_run.py`,
`scripts/inspect_source_data.py`.

Untracked and new: `scripts/base_task/tune_learning_rate.py`,
`scripts/evaluate_on_dev.py`, `findings/03_learning_rate_search.md`,
`Development/` (ignored).
