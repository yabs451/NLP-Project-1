# Interpretability of model collapse in induction circuits

Honours NLP research project. We train a small transformer on an in-context
learning task, then retrain it repeatedly on **its own predictions**, and watch
whether the *mechanism* inside it — the induction circuit — breaks down before
its accuracy does.

Built on [Singh et al. (2024), *What needs to go right for an induction
head?*](https://arxiv.org/abs/2404.07129) and their
[JAX/Equinox implementation](https://github.com/aadityasingh/icl-dynamics).

## The task

Each sequence is two symbol–label pairs followed by a query symbol matching one
of them. The model outputs the query's label, one of 5. Labels are reassigned at
random every sequence, so nothing can be memorised — the answer is only
available **from the context**. Symbols are precomputed ResNet18 feature vectors
of Omniglot characters.

The mechanism that solves this is an *induction circuit*: a previous-token head
in layer 0 lets each label token carry information about the symbol before it,
and an induction head in layer 1 lets the query attend to the label following
the matching symbol.

## What is done

| Stage | Status |
| --- | --- |
| Generation 0 — baseline trained on the real task | done (finding 01) |
| Generation 1 — first recursive successor, base task | done, no degradation this evaluator can resolve (finding 02) |
| Learning-rate search, 5 rates × 3 seeds | done, selected **1e-4** (finding 03) |
| Fresh 1,000-question re-evaluation and evaluator comparison | done (finding 03) |
| Further recursive generations | not run |
| Extended task (predict next symbol + its label) | **not implemented** |

Only the **base task** (predict the query's label) exists. The **extended task**
is future work and will live in `scripts/extended_task/`.

**Generation numbering.** Generation 0 is the original model trained on real
data; generation 1 is the first successor, trained on generation 0's predictions.
"N additional generations" means N successors; the total number of models is
N + 1.

**Two decisions are still open** — see the end of this file.

## Whose code is whose

- `upstream/icl-dynamics/` — **the authors' code, vendored unmodified** at commit
  `85b895c720844e795734b9391b32d2619065f9a1`, tracked directly in this repository
  (not a submodule), including the 16.6 MB Omniglot feature file. We never edit
  it, and we use only the first baseline in their `ih_paper_runs.sh`.
- `scripts/` — **our code**. It reads the authors' baseline command out of their
  shell script and calls their model, sampler, optimizer, loss, update,
  evaluation and checkpoint functions. We add only what they do not provide: our
  evaluation protocol, the recursive loop, the tuning search and the analysis.

## Folders

```
NLP-Project-1/
├── CLAUDE.md                 working conventions for this project
├── README.md                 this file
├── requirements.txt          the version pins we chose
├── requirements-lock.txt     every installed package, for exact rebuilds
├── scripts/                  maintained experiment and analysis code
├── results/                  numerical outputs, saved models, tables, figures
├── findings/                 written scientific interpretation
├── Development/              local, Git-ignored operational material (optional)
└── upstream/icl-dynamics/    the authors' code, unmodified
```

### Maintained scripts

| Script | What it does |
| --- | --- |
| `scripts/common.py` | Shared helpers: project paths, the authors' baseline arguments, loading features and checkpoints, rebuilding a run's evaluators, and scoring a checkpoint. Imported by everything else. |
| `scripts/prepare_evaluation_data.py` | Chooses the 100 development and 100 reserved final-test classes, checks they are disjoint, and builds both fixed development question sets. Run once, before any training. |
| `scripts/evaluate_on_dev.py` | Scores one saved checkpoint on a development set. |
| `scripts/base_task/train_original.py` | Trains one model on the original task by running the authors' `main.py`. |
| `scripts/base_task/tune_learning_rate.py` | Runs the learning-rate grid and applies the selection rule; `--compare-evaluators` re-scores the saved models on the smaller set and writes the comparison and figure. |
| `scripts/base_task/run_recursive.py` | Generates training data from a parent model's own predictions, then trains a successor on it. |
| `scripts/base_task/analyse_runs.py` | Learning curves, per-head previous-token and induction scores across checkpoints, an attention map, and final-checkpoint head ablations. |

`Development/` holds operational reports and debugging output. It is optional
local material: nothing under `scripts/` depends on it, and reproduction never
requires it.

## Setup

Python 3.10 (this machine has 3.10.11). All commands run **from the project
root** — the folder containing this README — in PowerShell. No environment
activation is needed; the commands call the interpreter directly.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --timeout 120 -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

Install from `requirements-lock.txt`; `requirements.txt` records the pins we
chose and why. JAX stays at 0.4.26, the version the authors tested. Everything
runs on **CPU** — JAX has no native-Windows GPU build — and needs no account.

## 1. Prepare the evaluation data

```powershell
.\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py
```

Run once before any training. Deterministic, so re-running is safe.

The authors' class split is 50 training / 1473 unused / 100 test, and their own
test evaluator is scored throughout training — so those 100 classes are not a
clean final test. We take both of our splits from the **1,473 classes no original
evaluator touches**: 100 for development and a disjoint 100 reserved as a final
test. This writes:

- `results/evaluation_data/class_splits.json` — exact class IDs, selection rule
  and every seed. **The one tracked file under `results/`**: it is the protocol
  and must stay auditable.
- `results/evaluation_data/eval_dev.h5` — 1,000 fixed development questions.
- `results/evaluation_data/eval_dev_large.h5` — 10,000 fixed development questions.

**Why two development sets.** Both are drawn from the same 100 held-out classes
under the same task rules, differing only in size and seed.

| | questions | seed | loaded into training? | used for |
| --- | ---: | ---: | --- | --- |
| `fsl_dev_class` | 1,000 | 1007 | yes | monitoring, scored at every evaluation point during a run |
| `fsl_dev_class_large` | 10,000 | 3007 | no | comparing final checkpoints of different models |

Near 97% accuracy the standard error is about 0.54 points at 1,000 questions and
0.17 at 10,000. The two sets are different samples, so their numbers are not
interchangeable — the same model can score 96.7% on one and 95.6% on the other
without contradiction.

**The reserved final test has no data generated and has never been scored.**
`--build-final-test` regenerates it deterministically when the project is ready.

## 2. Train one model on the original task

```powershell
.\.venv\Scripts\python.exe scripts/base_task/train_original.py --run-name my_run
```

Options: `--learning-rate` and `--init-seed` override the published values
(1e-5, seed 5); `--results-subfolder` places the run inside a subfolder of
`results/base_task/`; `--save-checkpoints endpoints` keeps only the first and
last checkpoint; `--dry-run` prints the underlying command and stops.

Always trains the authors' full schedule: 1,000,000 sequences = 31,250 updates
at batch size 32. The run refuses to write into an existing folder.

## 3. Reproduce the learning-rate search

```powershell
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py --dry-run
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py
```

`--dry-run` lists what would be trained, reused or skipped, then stops. The real
run trains the 5 × 3 grid (rates 1e-6, 3e-6, 1e-5, 3e-5, 1e-4; seeds 5, 6, 7) on
the original task with true targets, scores each final checkpoint on the
10,000-question set, and applies the selection rule.

Safe to stop and restart: results are written after every candidate and finished
candidates are skipped. A finished run whose configuration matches is reused
rather than retrained, which is why the published baseline serves as its own
grid cell.

## 4. Re-evaluate saved models and compare the evaluators

```powershell
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py --compare-evaluators
```

Loads each of the 15 saved final checkpoints and scores it **fresh** on the
1,000-question set — never reading a training log or an earlier record — then
reads back the recorded 10,000-question scores and writes the comparison and the
figure. No training.

To score a single saved checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_on_dev.py results/base_task/tuning/learning_rate_0.0001_init_seed_5
```

Add `--small` for the 1,000-question set, or `--checkpoint <sequences>` for a
checkpoint other than the last.

## 5. Run a recursive successor

```powershell
.\.venv\Scripts\python.exe scripts/base_task/run_recursive.py --parent results/base_task/<parent_run> --generations 1
```

Draws 1,000,000 questions from the **original** training distribution, asks the
parent for each answer and stores its **argmax over all five labels** as the new
target (replacing only the query's target), saves the dataset as class/exemplar
indices rather than feature vectors, then trains a **freshly initialised** model
— not the parent's weights — for exactly one pass. `--generations N` chains N
successors; completed generations are never retrained.

## 6. Analyse a run

```powershell
.\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py results/base_task/<run>
```

Writes `analysis/` inside the run folder: learning curves, per-head
previous-token and induction scores across the available checkpoints, an
attention map, and final-checkpoint head ablations. Candidate heads are picked
from **that model's own scores**, never inherited from another model. The script
adapts to whichever checkpoints exist.

## What the outputs contain, and how the analysis uses them

| File | Contents | Read by |
| --- | --- | --- |
| `config.json` | every resolved option the run used, including seeds and schedules | analysis, tuning, recursive pipeline |
| `log.h5` | one training loss and gradient norm per update (31,250), and per-evaluator accuracy/loss/in-context-accuracy arrays at each evaluation point | `analyse_runs.py`, for the learning curves |
| `checkpoints/` | one `.eqx` per checkpoint, named by sequence count, each ~809 KB holding weights, optimizer state and PRNG keys | analysis, evaluation, recursive pipeline |
| `analysis/` | `analysis.json` plus `curves.png`, `head_measures.png`, `attention_example.png` | findings 01 and 02 |
| `generated_training_data.h5` | successors only: class/exemplar indices and labels with the query target replaced, plus the true answers for diagnostics | the successor's own training loop |
| `generation_metadata.json` | successors only: generation number, parent checkpoint, generation rule, target-quality statistics, worked examples | finding 02 |
| `tuning/results.json` | the grid record and its 10,000-question scores | tuning, comparison |
| `tuning/selection.json` | selection rule, per-rate means, the winning rate and the chosen generation-0 checkpoint | finding 03 |
| `tuning/evaluator_comparison.json` | per-candidate scores on both evaluators, per-rate means, winner under each | finding 03 |
| `tuning/learning_rate_comparison.png` | the comparison figure | finding 03 |

We deliberately do **not** produce timing files, saved console transcripts,
standalone verification reports or setup-inspection dumps.

## Checkpoint policy

Main mechanistic runs save **55 checkpoints directly during training**:
initialisation, four early snapshots near 1,000 / 2,000 / 5,000 / 10,000
sequences, then every 20,000 through 1,000,000. Requests land on the next batch
boundary, so the early ones are saved at 1,024 / 2,016 / 5,024 / 10,016.

Tuning runs keep only the first and last checkpoint, because only final models
are compared there. Reducing snapshots does not reduce the learning curves,
which are logged separately in `log.h5`.

Generations 0 and 1 were trained before this policy, with 1,001 checkpoints
each. They now keep the 64 that the existing figures and this policy need, and
their analysis regenerates from them unchanged.

## Where the results are

- **Findings** (the science): `findings/01_baseline_induction_circuit.md`,
  `findings/02_first_recursive_generation.md`,
  `findings/03_learning_rate_search.md`.
- **Comparison table**: `results/base_task/tuning/evaluator_comparison.json`.
- **Comparison figure**: `results/base_task/tuning/learning_rate_comparison.png`.
- **Selected generation-0 checkpoint**:
  `results/base_task/tuning/learning_rate_0.0001_init_seed_5/checkpoints/00001000000.eqx`

## What is in Git

**Tracked:** all code, `CLAUDE.md`, `README.md`, both requirements files, the
whole vendored upstream repository, and `results/evaluation_data/class_splits.json`.

**Ignored:** `.venv/`, caches, `Development/`, and everything else under
`results/` — checkpoints, `log.h5`, generated datasets and evaluation data.

`findings/` tracking is an open decision and has not been set either way.

A fresh clone gives you the code, the authors' code, the feature data and the
evaluation protocol record. You must regenerate the environment, both evaluator
files and every run. Ignored files are **not backed up anywhere**.

## Open decisions and limitations

**Awaiting your decision:**

1. **Which development evaluator to keep.** Both currently exist. They select the
   same learning rate and give the same ranking, so the smaller one would have
   sufficed here; the larger one measures more precisely and would matter for
   closer comparisons. Retiring one means deleting its `.h5`, its build step in
   `prepare_evaluation_data.py`, its recorded seed, and the comparison code that
   reads both.
2. **Whether to retrain generation 0 at the selected 1e-4** before running the
   recursive chain, and whether `findings/` should be tracked.

**Limitations.** 1e-4 is the largest rate tested, so the optimum may lie above
it; the result is *best among the tested rates at this budget*. Three seeds, one
budget, final checkpoints only. No mechanistic analysis of the tuned models yet.
Generations 0 and 1 were trained at the authors' 1e-5, so they are a pilot rather
than part of the tuned lineage. Training settings must never drift between
generations in a comparison.

## Attribution

Task, model, samplers, training loop, evaluation and the progress measures are
from Singh et al. (2024), used under the terms of their repository. Our
contribution is the evaluation protocol, the recursive pipeline, the tuning
search, the analysis wrapper and the write-ups in `findings/`.
