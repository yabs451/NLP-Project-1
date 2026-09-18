# 07 — The five-model recursive experiment

16 September 2026. Local operational report; not tracked in Git.

## Existing files I used, and what they gave me

| File | What it does | What I took from it |
| --- | --- | --- |
| `CLAUDE.md` | Working conventions | The minimality rule, checkpoint policy, output policy and agent-output limits that shaped every decision below |
| `Development/reports/06_...md` | Previous stage's handover | That the 1e-5 lineage was already stripped, that the 1e-5/seed-5 tuning candidate had been rehomed, and the two open blockers |
| `results/base_task/tuning/selection.json` | Tuning outcome | The selected rate (1e-3), the predetermined seed (5) and the recorded accuracy (0.9960) I checked generation 0 against |
| `scripts/base_task/train_original.py` | Runs the authors' `main.py` | The trainer for generation 0, including the 55-checkpoint schedule |
| `scripts/base_task/run_recursive.py` | Generates data from a parent, trains a successor | The whole recursive loop, unchanged in substance |
| `scripts/base_task/analyse_runs.py` | Per-run analysis | Curves, previous-token and induction measures, attention map, ablations |
| `scripts/common.py` | Shared helpers | Checkpoint loading, evaluator rebuilding, scoring |
| `upstream/icl-dynamics/main_utils.py` | The authors' option validation | `check_opts`, which is what caught the bug below |

## What I changed, created or removed

**Changed**

- `scripts/base_task/train_original.py` — drops the authors' inherited
  `--ckpt_every` before applying our own schedule (see the bug below), and
  starts the training subprocess at Windows **AboveNormal** priority.
- `scripts/base_task/run_recursive.py` — successors now go to
  `results/base_task/recursive/generation_<n>/`, which removes the naming
  collision flagged in report 06 without any experiment-management machinery. It
  also raises its own process priority once at start.
- `scripts/common.py` — one small helper, `use_above_normal_priority()`.
- `scripts/base_task/analyse_runs.py` — added `--compare`, which reads each
  generation's existing `analysis.json` and writes the across-generation table
  and figure. It reads per-run analyses rather than recomputing, so every
  compared number came from identical code and identical checkpoint selection.
- `.gitignore` — `Development/`, `temporary_checks/` and `CLAUDE.md` are now
  ignored; all three were untracked with `git rm --cached`, local copies kept.
- `README.md` — rewritten as a public reproduction guide.
- `findings/03_learning_rate_search.md` → `findings/01_learning_rate_search.md`,
  with its references to the deleted findings and to the now-resolved
  "no intermediate checkpoints" limitation corrected.

**Created**

- `results/base_task/recursive/` — five generation folders plus the comparison
  table and figure.
- `findings/02_recursive_generations.md`.

**Removed**

- `results/base_task/generation_0_original_data_init_seed_5/` and
  `generation_1_generated_data_init_seed_5/` — the obsolete 1e-5 experiment.
- `findings/01_baseline_induction_circuit.md` and
  `02_first_recursive_generation.md` — the findings those runs supported,
  retired together with them rather than left pointing at deleted outputs.

Before removing the run folders I confirmed the only other references were those
two findings, and that the 1e-5/seed-5 tuning candidate is self-contained at
`results/base_task/tuning/learning_rate_1e-05_init_seed_5/`. After removal, all
18 tuning candidates still pass the configuration and completeness checks.

**Retained deliberately:** `evaluator_comparison.json` and
`evaluator_size_comparison.png` under `results/base_task/tuning/`, the historical
1,000-vs-10,000 evaluator check, labelled as covering the original 15 candidates
only. The 10,000-question evaluator was not rebuilt or run.

## Structure and tracking

A clone now receives only: `README.md`, `.gitignore`, both requirements files,
`scripts/`, `findings/`, `results/evaluation_data/class_splits.json`, and the
vendored `upstream/`. Everything else — `Development/`, `temporary_checks/`,
`CLAUDE.md` and the rest of `results/` — is local only. `findings/` tracking is
unchanged, as instructed. Nothing was committed, pushed or reset.

## The bug that stopped the first launch

The first chain failed within seconds. Upstream asserts *"At most one way of
ckpt iters should be specified"* (`main_utils.py:130`), and the authors'
baseline command already contains `--ckpt_every 1000`; appending `--ckpt_sched`
gave it two.

This had never been caught because stage 05 only ever verified the 55-checkpoint
schedule with `--dry-run`, which prints the command without running `main.py`.
The schedule had therefore never actually been executed. Fixed by deleting the
inherited option before adding ours, then verified by running upstream's own
`check_opts` on both the mechanistic and endpoints argument vectors. No partial
run was left behind: the `&&` chain stopped before any successor started, and the
empty generation-0 folder was deleted before relaunching.

## Experiment settings

Identical across all five models except the training targets: learning rate
1e-3, initialisation seed 5, batch size 32, 1,000,000 sequences = 31,250
updates, unchanged training-data seed, architecture and optimizer. Successors
start from fresh weights and a fresh optimizer. 55 checkpoints written directly
during training. Scored on the fixed 1,000-question development evaluator. The
reserved final test was not generated or scored.

## Results

Generation 0 agreed with the tuning candidate on both reported measures: dev
accuracy 0.9960, loss 0.0114. We did not compare the two models' parameters, so
this is agreement on what was reported, not established identity. No discrepancy
to report.

Generation 0 also scored 100% on `fsl_train`, the 1,000-sequence
training-distribution evaluator. That is a finite sample, not a measurement over
every question the distribution can produce.

On every one of the 1,000,000 generated examples, at every generation, the
parent's answer **matched the true answer**. The two paths differ in how targets
are obtained — generation 0 takes the sampler's label as each batch is drawn, a
successor reads a saved dataset labelled by the parent — but the resulting
target values were identical here. The questions and their order match by
construction (the successor draws from the same training key chain) plus the
earlier spot check against the authors' sampler; generation 0's data is never
written to disk, so no byte-level dataset comparison was possible.

What was compared between models: generation 0 vs generation 1 **final
parameters**, L2 difference exactly 0; and the **final and initial checkpoint
files** of generations 1–4, byte-identical to one another. The 55 intermediate
checkpoints were not compared. Every metric is flat:

| Gen | Dev acc | Dev loss | Induction | Prev-token | Ablate IH | Ablate PT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–4 | 0.9960 | 0.0114 | 0.9596 | 0.9999 | −0.3 pp | −2.7 pp |

Same heads selected in every generation (induction L1H2, previous-token L0H7),
chosen from each model's own scores rather than inherited.

## Interpretation and limitations

The chain was stable, so the experiment did not establish whether circuit
function weakens before accuracy: neither declined.

The explanation is specific to this configuration. Input construction, data
order, initialisation and optimisation were held fixed across generations, so the
only thing that could differ from generation 0 was the query targets — and those
matched the true answers on every generated example. With nothing varying,
training reached the same parameters. This does not show that label-only
recursion cannot degrade a model in general, nor predict what further
generations under these conditions would give.

Attention measurements from generation 0, identical in all five: seven of eight
layer-1 heads have a positive induction score, the strongest 0.9596, and two
layer-0 heads are near-maximal on the previous-token measure. Silencing the
strongest induction head changed development accuracy by 0.3 percentage points;
silencing the strongest previous-token head, by 2.7. Those are the effects of
those specific single-head interventions on this evaluator — not a measurement of
the circuit's overall importance, and not in themselves evidence of redundancy.
Compensation by other heads is one possible explanation and was not tested.

Limitations: one chain in which nothing varied; single-head zero-ablation on four
of sixteen heads, on one evaluator; 13 of the 55 checkpoints were analysed, and
the transition falls in a 40,000-sequence gap between analysed points, so no
ordering claim is possible; no independent replication.

## Requirements files

Both are fully pinned and agree exactly: `requirements.txt` lists 13 direct
packages with comments explaining the choices (JAX held at the authors' tested
0.4.26; matplotlib and tqdm only because the authors' analysis module imports
them), and the lock adds 15 transitive packages. No version differs between
them, and no command reads `requirements.txt` — the README installs from the
lock.

**Recommendation: consolidate to one pinned file.** The reasoning comments that
justify `requirements.txt` can live just as well in the merged file, attached to
the direct dependencies and marked as such, so nothing is lost. Two files only
create drift risk. Not done, as instructed.

## Unresolved

1. **What conditions to examine next.** We intend to discuss the extended task,
   where the model also predicts a following symbol and its label; because
   generated symbols can change the input distribution rather than only the
   targets, it offers a route to conditions where successors differ from their
   parent. Whether that produces drift is open. Any alternative condition needs
   its own scientific justification — degradation is not a required outcome.
2. Whether `findings/` should be tracked remains open.
