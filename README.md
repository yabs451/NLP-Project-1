# Interpretability of model collapse in induction circuits

Honours NLP research project. We train a small transformer on an in-context
learning task, then retrain it repeatedly on **its own predictions**, and watch
whether the *mechanism* inside it — the induction circuit — breaks down before
its accuracy does.

Built on [Singh et al. (2024), *What needs to go right for an induction
head?*](https://arxiv.org/abs/2404.07129) and their
[JAX/Equinox implementation](https://github.com/aadityasingh/icl-dynamics).

## What the task is

Each training sequence is two symbol–label pairs followed by a query symbol that
matches one of them. The model must output the query's label, from 5 possible
labels. Labels are reassigned at random for every sequence, so nothing can be
memorised: the answer is only available **from the context**. Symbols are
precomputed ResNet18 feature vectors of Omniglot characters.

The mechanism that solves this is an *induction circuit*: a previous-token head
in layer 0 lets each label token carry information about the symbol before it,
and an induction head in layer 1 lets the query attend to the label following
the matching symbol.

## What is done so far

| Stage | Status |
| --- | --- |
| Environment, data inspection, short trial | done (report 01) |
| Validation/final-test protocol | done (report 02) |
| Generation 0 — baseline trained on the real task | done, 96.7% on unseen classes (report 03, finding 01) |
| Generation 1 — first recursive successor, base task | done, 96.7%, no degradation (report 04, finding 02) |
| Further generations | not run — the pipeline supports them |
| Extended task (predict next symbol + its label) | **not implemented** |
| Hyperparameter tuning | deliberately deferred, see below |

The assignment requires two task variants. This repository currently implements
only the **base task** (predict the query's label). The **extended task**
(predict the label, the next symbol, and its label) is future work and will live
in `scripts/extended_task/`.

**Generation numbering.** Generation 0 is the original model trained on real
data. Generation 1 is the first successor, trained on generation 0's own
predictions. "N additional generations" means N successors; the total number of
models is N + 1. So far there are 2 models: generations 0 and 1.

## Whose code is whose

- `upstream/icl-dynamics/` — **the authors' code, vendored unmodified** at commit
  `85b895c720844e795734b9391b32d2619065f9a1`. It is tracked directly in this
  repository (not a Git submodule), including the 16.6 MB Omniglot feature file,
  so a clone gets everything. We never edit it. Their repository covers several
  papers; we use only the first baseline in `ih_paper_runs.sh`.
- `scripts/` — **our code**. It reads the authors' baseline command out of their
  shell script, calls their model, sampler, optimizer, loss, update, evaluation
  and checkpoint functions, and adds only what they do not provide: our
  evaluation protocol, the recursive loop, and the analysis.

## Folder structure

```
NLP-Project-1/
├── CLAUDE.md                     working conventions for this project
├── README.md                     this file
├── requirements.txt              the version pins we chose
├── requirements-lock.txt         every installed package, for exact rebuilds
├── scripts/
│   ├── common.py                 shared helpers: paths, checkpoints, evaluators
│   ├── prepare_evaluation_data.py  picks validation/final-test classes
│   ├── inspect_source_data.py    records the data, environment and provenance
│   ├── verify_run.py             checks a finished run is sound
│   └── base_task/                the label-only task
│       ├── train_original.py     trains generation 0
│       ├── run_recursive.py      generates data and trains successors
│       └── analyse_runs.py       curves, circuit measures, ablations
├── reports/                      operational handovers, numbered by stage
├── findings/                     the scientific record, plus small evidence
│   └── evidence/                 figures and JSON quoted by the findings
├── upstream/icl-dynamics/        the authors' code, unmodified
└── results/                      generated output (mostly not in Git)
    ├── evaluation_data/          class splits + the fixed development questions
    ├── base_task/                one folder per generation
    ├── protocol_checks/          evidence that protocol choice cannot affect training
    └── inspection/               data and environment record
```

Future extended-task code goes in `scripts/extended_task/`, with its runs under
`results/extended_task/`. Those folders do not exist yet, on purpose.

## Setup

Python 3.10 (this machine has 3.10.11). All commands below are run **from the
project root** (the folder containing this README), in PowerShell. No
environment activation is needed — the commands call the interpreter directly.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --timeout 120 -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` lists the pins we chose and why; `requirements-lock.txt` is
the full freeze and is what you should install from. JAX stays at 0.4.26, the
version the authors tested. matplotlib and tqdm are needed only because the
authors' analysis module imports them. Everything runs on **CPU** — JAX has no
native-Windows GPU build — and needs no online account.

Optional, records the data and environment for reproducibility:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_source_data.py
```

## 1. Prepare the evaluation data

```powershell
.\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py
```

Run this once, before any training. It is deterministic, so re-running is safe.

The authors' class split is 50 training / 1473 unused / 100 test, and their own
test evaluator is scored throughout training — so those 100 classes are not a
clean final test. We instead take both of our splits from the **1,473 classes no
original evaluator touches**: 100 for development validation and a disjoint 100
reserved as a final test. The script checks disjointness from each other, from
the training classes, and from the authors' test classes, then writes:

- `results/evaluation_data/class_splits.json` — the exact class IDs, the
  selection rule and every seed. **This file is tracked in Git**: it is the
  protocol, and it must stay auditable.
- `results/evaluation_data/eval_dev.h5` — 1,000 fixed development questions.

**The reserved final test has no data generated and has never been scored.**
`--build-final-test` regenerates it deterministically when the project is ready.

## 2. Train generation 0 (the original baseline)

```powershell
.\.venv\Scripts\python.exe scripts/base_task/train_original.py --full --protocol assignment --run-name generation_0_original_data_init_seed_5
```

About 5.6 minutes on CPU and ~795 MB of checkpoints.

- `--full` uses the authors' full schedule: 1,000,000 sequences = 31,250 updates
  at batch size 32. Omit it for a 3,200-sequence smoke test.
- `--protocol assignment` scores our development classes and **does not run** the
  authors' test-class evaluator. `--protocol reproduction` (the default) runs
  their original four evaluators instead.
- `--dry-run` prints the underlying command without training.

Protocol choice affects evaluation only, never training.

## 3. Run one recursive successor

```powershell
.\.venv\Scripts\python.exe scripts/base_task/run_recursive.py --parent results/base_task/generation_0_original_data_init_seed_5 --generations 1
```

About 30 seconds to generate the data and 8 minutes to train, plus ~810 MB of
checkpoints and a 4.3 MB dataset.

For each successor this: draws 1,000,000 questions from the **original** training
distribution (same 50 classes, exemplar 0, same context construction and label
pairs); asks the parent for each answer and stores its **argmax over all five
labels** as the new target, replacing only the query's target; saves the dataset
as class/exemplar/label indices rather than copying feature vectors; then trains
a **freshly initialised** model — not the parent's weights — for exactly one pass.

## 4. Continue for more generations

Point `--parent` at the newest generation. Nothing needs editing:

```powershell
.\.venv\Scripts\python.exe scripts/base_task/run_recursive.py --parent results/base_task/generation_1_generated_data_init_seed_5 --generations 3
```

That trains generations 2, 3 and 4 in sequence, each from the one before.
Completed generations are never retrained. Budget roughly 9 minutes and 810 MB
per generation.

## 5. Verify and analyse

```powershell
.\.venv\Scripts\python.exe scripts/verify_run.py results/base_task/generation_1_generated_data_init_seed_5
.\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py results/base_task/generation_1_generated_data_init_seed_5
```

`verify_run.py` (about 10 seconds) checks the final checkpoint loads, the optimizer did
the expected 31,250 updates, parameters are finite and moved, the training log is
complete, the query's target never reached the model input, and that re-scoring
the final checkpoint reproduces the metrics the run logged.

`analyse_runs.py` (about 11 seconds) writes `analysis/` inside the run folder:
learning curves, per-head previous-token and induction scores across 15
checkpoints, an attention map, and final-checkpoint head ablations. Candidate
heads are picked from **that model's own scores**, never inherited from another
model.

Read the science in `findings/`, and the operational detail in `reports/`.

## What validation does, and when

| Evaluator | Data | When it runs | What it tells you |
| --- | --- | --- | --- |
| `fsl_dev_class` | 1,000 fixed sequences, 100 **unseen** classes | every 5,000 sequences | **our validation**: generalisation to new characters |
| `fsl_train` | training distribution | every 5,000 sequences | diagnostic only — *not* guaranteed disjoint from training |
| `fsl_val_rl` | training classes, unseen label pairings | every 5,000 sequences | diagnostic |
| `fsl_train_valex` | training classes, unseen exemplars | every 5,000 sequences | diagnostic |
| reserved final test | 100 further disjoint classes | **never run** | held back until the project is finished |

Every evaluator's questions are drawn once and reused for the whole run, so
every checkpoint is scored on identical questions. Accuracy is always argmax;
`in_context_acc` restricts the argmax to the two labels present in context.
Chance is 20% for accuracy and 50% for context-restricted accuracy.

For a successor, **training loss is measured against the parent's generated
targets**, while development loss is against the true answers. They are not
comparable to each other.

## What the generated files contain

| File | Contents |
| --- | --- |
| `config.json` | every resolved option the run used, including seeds and schedules |
| `command.json` | the exact argument vector (generation 0 only, which runs as a subprocess) |
| `log.h5` | raw metrics: one training loss and gradient norm per update (31,250), and per-evaluator `acc`/`loss`/`in_context_acc`/`prob` arrays of 1,000 values at each of 201 evaluation points |
| `checkpoints/` | one `.eqx` file per checkpoint, named by sequence count (`00001000000.eqx`), each ~809 KB holding model weights, optimizer state and PRNG keys. 1,001 files per full run, ~800 MB |
| `generated_training_data.h5` | successors only: `class_idxs`, `exemplar_idxs`, `labels` (query target replaced by the parent's answer) and `true_query_label` for diagnostics. Indices, not feature vectors, so 1,000,000 examples fit in 4.3 MB |
| `generation_metadata.json` | successors only: generation number, parent checkpoint, generation rule, seeds, target-quality statistics and worked examples |
| `timing.json` | measured durations |
| `verification.json` | output of `verify_run.py` |
| `analysis/` | `analysis.json` plus `curves.png`, `head_measures.png`, `attention_example.png` |

## What is in Git and what is not

**Tracked:** all code, `CLAUDE.md`, `README.md`, both requirements files, every
report and finding, the small evidence in `findings/evidence/`, the whole
vendored upstream repository, and `results/evaluation_data/class_splits.json`.

**Ignored:** `.venv/`, caches, and everything else under `results/` —
checkpoints (~800 MB per run), `log.h5`, generated datasets and evaluation data.

**A fresh clone therefore gives you** the code, the authors' code, the feature
data, the evaluation protocol record, and every report and finding with its
numbers and figures. **You must regenerate**: the virtual environment,
`eval_dev.h5`, and all training runs. Every number a report or finding relies on
is quoted in that document, precisely because the runs are not in Git.

Ignored files are **not backed up anywhere**. Do not assume a run can be
recovered; re-run it.

## Limitations

- One initialisation seed (5) throughout. No multi-seed robustness.
- We have **not** reproduced the paper's results: no published number was ever
  compared against.
- Two generations cannot show whether circuit function degrades before accuracy.
  That needs a longer chain.
- Generation 1 reuses generation 0's question stream, which maximises control
  but makes the two runs statistically dependent.
- Head ablation is single-head, zero-ablation only, on four of sixteen heads.

**Hyperparameter tuning is deliberately deferred.** It belongs after this pilot
and before the main controlled comparison. If training settings change, the
baseline must be retrained to match — settings must never drift between
generations, or the comparison is meaningless.

## Attribution

Task, model, samplers, training loop, evaluation and the progress measures are
from Singh et al. (2024), used under the terms of their repository. Our
contribution is the evaluation protocol, the recursive pipeline, the analysis
wrapper, and the scientific write-ups in `findings/`.
