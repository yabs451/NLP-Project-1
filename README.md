# Interpretability of model collapse in induction circuits

An Honours NLP research project. We train a small transformer on an in-context
learning task, then retrain it repeatedly on **its own predictions**, and ask a
mechanistic question:

> Does the induction circuit inside the model weaken across recursive
> generations *before* its overall accuracy declines?

This is an initial investigation over one chain of five models. Degradation is
not assumed.

Built on [Singh et al. (2024), *What needs to go right for an induction
head?*](https://arxiv.org/abs/2404.07129) and their
[JAX/Equinox implementation](https://github.com/aadityasingh/icl-dynamics).

## The task

Each sequence is two symbol–label pairs followed by a query symbol matching one
of them. The model outputs the query's label, one of five. Labels are reassigned
at random for every sequence, so no fixed symbol-to-label mapping can be relied
on — the answer has to be read out of the context. Symbols are precomputed
ResNet18 feature vectors of Omniglot characters.

The mechanism that solves this is an *induction circuit*: a previous-token head
in layer 0 lets each label token carry information about the symbol before it,
and an induction head in layer 1 lets the query attend to the label that follows
the matching symbol.

## What is implemented

- The **base task** only: the model predicts the query's label. The extended
  task (also predicting a following symbol and its label) is not implemented.
- A **learning-rate search** over six rates × three initialisation seeds.
- A **five-model recursive chain**: generation 0 trained on the real task, then
  generations 1–4 each trained on the previous generation's own answers.
- Analysis of accuracy, loss, previous-token and induction attention measures,
  head ablations, and a comparison across the five generations.

Generation 0 is the original model. "N additional generations" means N
successors, so five models means generation 0 plus four successors.

## Whose code is whose

- `upstream/icl-dynamics/` — **the authors' code, vendored unmodified** at commit
  `85b895c720844e795734b9391b32d2619065f9a1`, tracked here directly (not a
  submodule) including the 16.6 MB Omniglot feature file. We never edit it, and
  use only the first baseline configuration in their `ih_paper_runs.sh`.
- `scripts/` — **our code**. It reads the authors' baseline command out of their
  shell script and calls their model, sampler, optimizer, loss, update,
  evaluation and checkpoint functions. We add only what they do not provide: our
  evaluation protocol, the recursive loop, the tuning search and the analysis.

## Structure

```
NLP-Project-1/
├── README.md
├── requirements.txt          version pins, with the reasoning
├── requirements-lock.txt     full freeze — install from this
├── scripts/
│   ├── common.py                    shared helpers
│   ├── prepare_evaluation_data.py   class splits + the development questions
│   ├── evaluate_on_dev.py           score one saved checkpoint
│   └── base_task/
│       ├── train_original.py        train a model on the real task
│       ├── tune_learning_rate.py    the learning-rate search
│       ├── run_recursive.py         generate data from a parent, train successors
│       └── analyse_runs.py          per-run analysis and the cross-generation comparison
├── findings/                 the scientific write-ups
├── results/                  everything the scripts generate (not distributed)
└── upstream/icl-dynamics/    the authors' code, unmodified
```

| Script | What it does |
| --- | --- |
| `common.py` | Project paths, the authors' baseline arguments, loading features and checkpoints, rebuilding a run's evaluators, scoring a checkpoint. Imported by the rest. |
| `prepare_evaluation_data.py` | Picks the 100 development and 100 reserved final-test classes, checks they are disjoint, and builds the fixed 1,000-question development set. Run once. |
| `evaluate_on_dev.py` | Scores one saved checkpoint on the development set. |
| `train_original.py` | Trains one model on the real task by running the authors' `main.py`, with our evaluator set and checkpoint schedule. |
| `tune_learning_rate.py` | Trains the learning-rate grid, applies the selection rule, writes the results table, selection record and figure. |
| `run_recursive.py` | For each successor: generates a million training examples from the parent's own answers, then trains a freshly initialised student on them. |
| `analyse_runs.py` | Per run: learning curves, per-head attention measures across checkpoints, an attention map, head ablations. With `--compare`: the across-generation table and figure. |

## Setup

Python 3.10. All commands run **from the project root** in PowerShell; no
environment activation is needed because they call the interpreter directly.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --timeout 120 -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

Everything runs on **CPU** — JAX has no native-Windows GPU build — and needs no
online account. JAX stays at 0.4.26, the version the authors tested.

## Reproducing the project

### 1. Build the evaluation data

```powershell
.\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py
```

Run once, before any training. Deterministic, so re-running is safe.

The authors' class split is 50 training / 1473 unused / 100 test, and their own
test evaluator is scored throughout training, so those 100 classes cannot serve
as a clean final test. We take both of our splits from the **1,473 classes no
original evaluator touches**: 100 for development and a disjoint 100 reserved as
a final test.

This writes `results/evaluation_data/class_splits.json` (the exact class IDs,
selection rule and seeds) and `results/evaluation_data/eval_dev.h5` — the fixed
**1,000-question development evaluator**, the single evaluator used everywhere
in this project. **The reserved final test has never been generated or scored.**

### 2. Reproduce the learning-rate search

```powershell
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py
```

Trains six rates (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3) at three initialisation
seeds (5, 6, 7) — 18 models — on the real task, scores each final checkpoint on
the development set, and selects the rate with the highest mean accuracy across
the three seeds.

**Selected: 1e-3**, the best *among the tested rates at this training budget* —
not a proven global optimum. It sits at the top of the range tested, and the
search stopped there.

Results are written after every candidate and finished candidates are skipped,
so the command is safe to stop and restart.

### 3. Train generation 0

```powershell
.\.venv\Scripts\python.exe scripts/base_task/train_original.py --results-subfolder recursive --run-name generation_0 --learning-rate 0.001 --init-seed 5
```

Seed 5 was fixed in advance as the seed carried forward; it was not chosen for
scoring highest.

### 4. Train the four successors

```powershell
.\.venv\Scripts\python.exe scripts/base_task/run_recursive.py --parent results/base_task/recursive/generation_0 --generations 4
```

For each successor this draws 1,000,000 questions from the **original** training
distribution — same classes, same context construction, correct context labels —
asks the parent for each answer and keeps its **argmax over all five labels** as
the training target, replacing only the query's answer. No true targets are
mixed in. It then trains a **freshly initialised** model (new weights, new
optimizer — not the parent's weights) for one pass. `--generations N` chains N
successors; finished generations are never retrained.

### 5. Analyse

```powershell
.\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py results/base_task/recursive/generation_0
.\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py --compare results/base_task/recursive
```

Run the first command once per generation, then the comparison. Per run it
writes learning curves, per-head previous-token and induction scores across
checkpoints, an attention map and final-checkpoint head ablations. The
comparison writes the across-generation table and figure.

Candidate heads are selected from **each model's own final scores**, never
inherited from another generation; the comparison records whether the same head
indices came out anyway.

## What gets generated

`results/` is produced by the commands above and is not distributed with the
repository — regenerate it by following the steps in order.

| Path | Contents |
| --- | --- |
| `results/evaluation_data/` | `class_splits.json` (the evaluation protocol) and `eval_dev.h5` (the 1,000 fixed questions) |
| `results/base_task/tuning/` | one folder per tuning candidate, plus `results.json` (all 18 candidates), `selection.json` and `learning_rate_comparison.png` |
| `results/base_task/recursive/generation_<n>/` | one folder per generation |
| `results/base_task/recursive/generation_comparison.json` / `.png` | the across-generation comparison |

Inside a generation folder:

| File | Contents |
| --- | --- |
| `config.json` | every resolved setting, including seeds and schedules |
| `log.h5` | one training loss and gradient norm per update (31,250), plus development accuracy and loss at each evaluation point |
| `checkpoints/` | the saved models, named by sequence count |
| `generated_training_data.h5` | successors only: the training set as class and exemplar indices plus labels — compact, not copied feature vectors |
| `generation_metadata.json` | successors only: parent checkpoint, generation rule, how good the parent's answers were, and worked examples |
| `analysis/` | `analysis.json` plus `curves.png`, `head_measures.png`, `attention_example.png` |

Two older files under `results/base_task/tuning/` — `evaluator_comparison.json`
and `evaluator_size_comparison.png` — are retained history from a one-off check
of whether 1,000 development questions were enough to choose between models.
They cover the **original 15 candidates only**, not all 18.

## Checkpoint policy

Runs that will be analysed mechanistically save **55 checkpoints, written
directly during training**: the initialisation, four early snapshots while the
circuit is still forming, then every 20,000 sequences to 1,000,000. Requests
land on the next batch boundary, so the early ones are saved at 1,024 / 2,016 /
5,024 / 10,016.

Tuning candidates instead save only the **first and last** checkpoint, because
only their final models are ever compared. That keeps the 18-model search small.
It also means a tuning candidate cannot support mechanistic analysis, which is
why generation 0 is trained separately rather than reused from the search.

Reducing snapshots never reduces the learning curves: those come from `log.h5`,
which is written at every evaluation point regardless.

## Limitations

- **One chain, controlled seeds.** Five models in a single recursive line, all
  sharing an initialisation seed and training-data seed. That isolates the
  effect of the changing targets, but it is not a sample of independent chains.
- **One learning rate, one budget.** 1e-3 was the best of six rates at 31,250
  updates. It sits at the edge of the tested range.
- **Attention patterns are not causal claims.** A high induction score for a
  head does not establish that the head is causally important. Single-head
  ablation measures the effect of that one intervention on one evaluator; it
  does not measure the importance of the circuit as a whole.
- **Degradation across generations and development within training are separate
  questions.** If two measures move together, nothing here can say which moved
  first.
- The reserved 100 final-test classes have never been scored.

## Attribution

The task, model, samplers, training loop, evaluation and the progress-measure
definitions are from Singh et al. (2024), used under the terms of their
repository. Ours is the evaluation protocol, the recursive pipeline, the tuning
search, the analysis and the write-ups in `findings/`.
