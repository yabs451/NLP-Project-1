# 06 — Retirements and the 1e-3 learning-rate extension

16 September 2026. Local operational report; not tracked in Git.

Scope: act on the user's four decisions — keep the 1,000-question evaluator,
add learning rate 1e-3 at seeds 5/6/7, retire the 1e-5 generation-0/1
experiment, and keep the retired 10,000-question functionality in a trackable
`temporary_checks/`. No recursive successor was run and the extended task was
not implemented.

## Files used and changed

| File | Change |
| --- | --- |
| `scripts/common.py` | Dropped `LARGE_DEV_EVALUATOR_FILE`; one development evaluator remains |
| `scripts/prepare_evaluation_data.py` | Builds only the 1,000-question set; the 10,000-question build, its seed record and `LARGE_DEV_SEQUENCES` removed. The 1,000-question build path is untouched |
| `scripts/evaluate_on_dev.py` | `--small` removed; always scores the 1,000-question set |
| `scripts/base_task/tune_learning_rate.py` | Grid extended to six rates; scores on the 1,000-question set; `--compare-evaluators` and `compare_evaluators()` removed; two-evaluator plot replaced by a single-evaluator `plot_summary`, now written by the ordinary run |
| `README.md`, `CLAUDE.md` | Single-evaluator workflow, `temporary_checks/`, retired runs, new grid and winner |
| `findings/01`, `findings/02` | Retirement notes added at the top |
| `findings/03_learning_rate_search.md` | Rewritten for six rates; the evaluator-size comparison kept as a scoped historical note |
| `temporary_checks/` | New: retired evaluator rebuild script and its README |

Upstream untouched. No timing collection, console capture, monitoring file or
verification report was added.

## What was retired or moved, and why

**The 10,000-question evaluator.** Removed from the main workflow. Its `.h5`
(~60 MB) was deleted rather than stored, because it is exactly regenerable:
`temporary_checks/rebuild_large_dev_evaluator.py` recreates it from the recorded
seed 3007 and the unchanged development classes. That script imports the
maintained helpers in `scripts/`; nothing under `scripts/` imports it.
`temporary_checks/` is deliberately **not** in `.gitignore`.

The comparison measurements stayed under `results/` with historical labels —
`evaluator_comparison.json` and `evaluator_size_comparison.png` (renamed from
`learning_rate_comparison.png`, which is now the six-rate figure). Both are
marked as covering the original 15 candidates only; the 1e-3 candidates were
never scored on 10,000 questions.

**The 1e-5 generation-0/1 experiment.** The dependency check found that the
(1e-5, seed 5) tuning candidate pointed at the generation-0 run folder. Before
removing anything, the minimal candidate files — `config.json` and the final
checkpoint — were copied into the normal tuning layout at
`results/base_task/tuning/learning_rate_1e-05_init_seed_5/`, and every reference
in `results.json`, `selection.json` and `evaluator_comparison.json` was
repointed. Verified afterwards that the candidate still passes the configuration
check and counts as complete. `log.h5` and the other 63 checkpoints were not
copied: nothing reads them for tuning.

Both retired run folders then had their `checkpoints/`, `log.h5` and generated
dataset deleted. Their `analysis/` outputs and `generation_metadata.json` were
kept because findings 01 and 02 cite those figures and numbers directly —
removing them would have broken two retained findings. Each folder is now
~424 KB. A retirement note at the top of each finding states plainly that the
numbers remain valid but can no longer be regenerated without retraining.

`results/` went from 497 MB to 358 MB across this stage (and from 1.8 GB before
the previous stage's pruning).

**Scores migrated, not recomputed.** The 15 completed candidates already had
fresh 1,000-question scores in `evaluator_comparison.json`. Those values were
copied into `results.json` so the tuning driver skipped them as completed. No
model was retrained and no finished model was re-evaluated.

## The three new candidates

Trained sequentially through the existing driver and run organisation, at 1e-3
with seeds 5, 6, 7: original training distribution with true query targets,
fresh weights and optimizer, batch size 32, 1,000,000 sequences = 31,250
updates, same training-data seed, endpoints checkpoint policy, final checkpoint
scored on the 1,000-question set.

| seed | accuracy | loss |
| --- | ---: | ---: |
| 5 | 0.9960 | 0.0114 |
| 6 | 0.9980 | 0.0088 |
| 7 | 0.9980 | 0.0096 |
| **mean** | **0.9973** | **0.0099** |

All three completed; none diverged and no loss was non-finite. All 18 grid
records are `completed`.

## Selection

Applying the unchanged rule to all six rates on the 1,000-question evaluator:

| rate | mean accuracy | mean loss |
| --- | ---: | ---: |
| 1e-6 | 0.5157 | 0.7593 |
| 3e-6 | 0.8127 | 0.4135 |
| 1e-5 | 0.9677 | 0.0928 |
| 3e-5 | 0.9790 | 0.0822 |
| 1e-4 | 0.9923 | 0.0278 |
| **1e-3** | **0.9973** | **0.0099** |

**Selected: 1e-3**, best among the tested rates at this training budget, again at
the upper boundary. The search stopped there by decision; 3e-4 and rates above
1e-3 were not tested.

Future generation-0 parent, seed 5 as predetermined (not the best seed — 6 and 7
both scored 0.998):

`results/base_task/tuning/learning_rate_0.001_init_seed_5/checkpoints/00001000000.eqx`

## Checks

| Check | Result |
| --- | --- |
| Processes before editing | No training running; no watcher started this stage |
| 1e-5/seed-5 candidate after migration | Configuration matches, counts complete |
| No reference left to the retired run folder | Confirmed across all three tuning records |
| 1,000-question evaluator unchanged | Build path untouched (seed 1007, `opts.eval_iters`); `prepare_evaluation_data.py` deliberately not re-run while training read the file |
| Grid completeness | 18/18 completed, none diverged |
| Figure | Regenerated by the ordinary tuning run, six rates |
| Document references | Every `results/`, `scripts/`, `temporary_checks/` path cited by README and findings exists |
| Retired references in maintained code | None remain |
| Reserved final test | Not generated, not scored |

## Limitations

- **1e-3 is at the grid boundary** and the curve is flattening rather than still
  climbing; a better rate above it cannot be ruled out.
- **The selected model has only two checkpoints** (first and last). The
  mechanistic analysis used in findings 01 and 02 needs snapshots across
  training, so it cannot run on this model as it stands.
- Three seeds, one budget, final checkpoints only. No mechanistic analysis at
  1e-3.
- Findings 01 and 02 remain readable but are no longer regenerable.

## Ready for the six-successor task, and what is not

Ready: the recursive pipeline (`run_recursive.py`) is unchanged and working; the
evaluation protocol, class splits and 1,000-question evaluator are fixed; the
parent checkpoint is selected and verified present.

**Two things to settle first:**

1. **Whether to retrain generation 0 at 1e-3 under the 55-checkpoint policy.**
   Without it there is no mid-training mechanistic data for the tuned lineage,
   which is what the project's central question needs. Roughly 10–15 minutes.
2. **Successor folder naming.** `run_recursive.py` derives the folder from the
   generation number and seed only, ignoring the parent, so two lineages would
   want the same name. It refuses to overwrite, so it fails safely, but the
   naming scheme needs deciding before a six-generation chain is launched.

## Git status

Nothing committed, pushed or reset; no remote changed; nothing staged. HEAD
remains `ed6d2ba`. `temporary_checks/` is untracked but deliberately not
ignored, awaiting the user's decision to commit it.
