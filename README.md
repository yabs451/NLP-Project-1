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
| Learning-rate search, 6 rates × 3 seeds | done, selected **1e-3** (finding 03) |
| Pilot generation 0 and generation 1 at 1e-5 | **retired** — the search rejected that rate (findings 01, 02) |
| Recursive generations at the selected rate | not run |
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
├── temporary_checks/         one retired comparison, kept at the user's request
├── Development/              local, Git-ignored operational material (optional)
└── upstream/icl-dynamics/    the authors' code, unmodified
```

### Maintained scripts

| Script | What it does |
| --- | --- |
| `scripts/common.py` | Shared helpers: project paths, the authors' baseline arguments, loading features and checkpoints, rebuilding a run's evaluators, and scoring a checkpoint. Imported by everything else. |
| `scripts/prepare_evaluation_data.py` | Chooses the 100 development and 100 reserved final-test classes, checks they are disjoint, and builds the fixed 1,000-question development set. Run once, before any training. |
| `scripts/evaluate_on_dev.py` | Scores one saved checkpoint on the development set. |
| `scripts/base_task/train_original.py` | Trains one model on the original task by running the authors' `main.py`. |
| `scripts/base_task/tune_learning_rate.py` | Runs the learning-rate grid, applies the selection rule, and writes the results table, the selection record and the comparison figure. |
| `scripts/base_task/run_recursive.py` | Generates training data from a parent model's own predictions, then trains a successor on it. |
| `scripts/base_task/analyse_runs.py` | Learning curves, per-head previous-token and induction scores across checkpoints, an attention map, and final-checkpoint head ablations. |

`Development/` holds operational reports and debugging output. It is optional
local material: nothing under `scripts/` depends on it, and reproduction never
requires it.

`temporary_checks/` holds a **retired evaluation-size comparison**, kept at the
user's request. The project briefly maintained a second, 10,000-question
development evaluator to check whether 1,000 questions were precise enough to
choose between models. They agreed, so the larger set was retired. Nothing in
the main experiment uses or imports it — see `temporary_checks/README.md`.

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
- `results/evaluation_data/eval_dev.h5` — 1,000 fixed development questions
  (`fsl_dev_class`, seed 1007). This is the project's single development
  evaluator: it is loaded into training for monitoring, and it is what final
  models are compared on.

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
run trains the 6 × 3 grid (rates 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3; seeds 5, 6,
7) on the original task with true targets, scores each final checkpoint on the
1,000-question development set, and applies the selection rule. It also writes
the comparison figure.

1e-3 was added after the first five rates put the winner at the top of the
range. The search stopped there — no further rate was tested.

Safe to stop and restart: results are written after every candidate and finished
candidates are skipped. A finished run whose configuration matches is reused
rather than retrained, which is why the published baseline serves as its own
grid cell.

## 4. Score a single saved checkpoint

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_on_dev.py results/base_task/tuning/learning_rate_0.001_init_seed_5
```

Add `--checkpoint <sequences>` for a checkpoint other than the last.

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
| `tuning/results.json` | every candidate's learning rate, seed, run path and 1,000-question accuracy and loss | tuning, finding 03 |
| `tuning/selection.json` | selection rule, per-rate means, the winning rate and the chosen generation-0 checkpoint | finding 03 |
| `tuning/learning_rate_comparison.png` | final accuracy against learning rate, all six rates | finding 03 |
| `tuning/evaluator_comparison.json`, `tuning/evaluator_size_comparison.png` | **historical**: the retired 1,000-vs-10,000 comparison, covering the original 15 candidates only | finding 03 |

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

The retired 1e-5 pilot runs (generations 0 and 1) no longer hold checkpoints at
all: only the analysis outputs that findings 01 and 02 cite were kept, so those
numbers and figures remain readable but cannot be regenerated without
retraining.

## Where the results are

- **Findings** (the science): `findings/01_baseline_induction_circuit.md`,
  `findings/02_first_recursive_generation.md`,
  `findings/03_learning_rate_search.md`.
- **Tuning table**, all 18 candidates: `results/base_task/tuning/results.json`.
- **Selection record**: `results/base_task/tuning/selection.json`.
- **Comparison figure**: `results/base_task/tuning/learning_rate_comparison.png`.
- **Selected model** (learning rate 1e-3, seed 5):
  `results/base_task/tuning/learning_rate_0.001_init_seed_5/checkpoints/00001000000.eqx`

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

**Before the six-successor experiment:**

1. **The selected model has only its first and last checkpoint.** Tuning runs use
   the endpoints policy, so the 1e-3 / seed-5 model cannot support the
   mechanistic analysis in findings 01 and 02, which needs snapshots across
   training. Producing those would mean retraining at 1e-3 under the
   55-checkpoint policy. Not done here, and not done silently.
2. **Successor folder naming.** `run_recursive.py` names a successor from the
   generation number and seed alone, ignoring which parent it came from, so two
   lineages would collide. It fails safely rather than overwriting, but the
   naming needs settling before six successors are run.
3. **Whether `findings/` should be tracked** — still undecided, untouched.

**Limitations.** 1e-3 is the largest rate tested and the search stopped there by
decision, so the optimum may lie above it; the result is *best among the tested
rates at this budget*. Three seeds, one budget, final checkpoints only. No
mechanistic analysis at 1e-3 yet. Training settings must never drift between
generations in a comparison.

## Attribution

Task, model, samplers, training loop, evaluation and the progress measures are
from Singh et al. (2024), used under the terms of their repository. Our
contribution is the evaluation protocol, the recursive pipeline, the tuning
search, the analysis wrapper and the write-ups in `findings/`.
