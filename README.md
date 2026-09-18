# Interpretability of model collapse in induction circuits

An Honours NLP research project. We train a small transformer on an in-context
learning task, then retrain it repeatedly on **its own predictions**, and ask two
questions:

> How does the way a parent turns its predictions into training targets — the
> synthetic-label generation strategy — affect performance degradation and
> induction-circuit function across recursive generations?

> Do attention-pattern measures, measured head-ablation effects and predictive
> performance change on the same schedule, or on different ones?

Each condition is a single chain of five models from a shared parent, not a set
of independent replications. Degradation is not assumed.

Built on [Singh et al. (2024), *What needs to go right for an induction
head?*](https://arxiv.org/abs/2404.07129) and their
[JAX/Equinox implementation](https://github.com/aadityasingh/icl-dynamics).

## The two tasks

Each sequence starts the same way: two symbol–label pairs, then a query symbol
matching one of them. Labels are reassigned at random for every sequence, so no
fixed symbol-to-label mapping can be relied on — the answer has to be read out of
the context. Symbols are precomputed ResNet18 feature vectors of Omniglot
characters.

**Base task.** The model predicts the query's label, one of five.

**Extended task.** The model predicts three things in turn: the query's label, a
*next symbol* chosen from the two already in the context, and that symbol's
label. For a context `A → 1, B → 4` with query `A`, a valid continuation is
`1, B, 4`. Either context symbol is a legitimate choice, so only the two label
predictions have a single right answer.

Restricting the next symbol to the two context symbols is our design choice, not
a requirement of the task definition. It keeps the symbol output a two-way
decision that reuses the existing symbol features.

The mechanism that solves this is an *induction circuit*: a previous-token head
in layer 0 lets each label token carry information about the symbol before it,
and an induction head in layer 1 lets the query attend to the label that follows
the matching symbol.

## What is implemented

- A **learning-rate search for the base task**, over six rates × three
  initialisation seeds.
- A **base-task recursive chain** of five models at the selected rate:
  generation 0 trained on the real task, then generations 1–4 each trained on
  the previous generation's own answers.
- A **separate learning-rate search for the extended task**, over the same six
  rates × three seeds, selecting the rate used by everything below.
- **Three families of extended-task recursive chains** at that selected rate,
  nine conditions in total. Every chain is five models, and **all of them share
  one generation 0**. Each family changes exactly one thing from a shared
  reference condition — original opening questions, argmax labels, next symbol
  sampled at temperature 1:
  - **label generation** — argmax (the reference), or label sampling at
    temperature 1, 3 or 5;
  - **context feedback** — each child's opening questions are built from the
    frequencies of the symbols its parent actually generated, sharpened at
    context temperature 1, 1/3 or 0.2;
  - **next-symbol temperature** — the parent picks between the two context
    symbols at temperature 1/3 or 0.2 instead of 1.

  Most chains are five models (generations 0–4). The four sharpened symbol
  conditions run to **generation 6**, seven models each.
- Analysis of accuracy and loss at each output position, teacher-forced and
  self-generated continuations, the corruption of the generated targets,
  next-symbol position and identity behaviour, previous-token and induction
  attention measures, single-head ablations, within-training curves, and
  comparisons within and across conditions.

Generation 0 is the original model. "N additional generations" means N
successors, so five models means generation 0 plus four successors.

## The write-ups

`findings/` holds the scientific record, numbered in the order the work was done:

| Finding | Question |
| --- | --- |
| `01_learning_rate_search.md` | Which learning rate suits the **base** task, over six rates × three seeds? |
| `02_recursive_generations.md` | What happens to the base task across five recursive generations? |
| `04_extended_task_tuning.md` | Which learning rate suits the **extended** task, over the same grid? |
| `05_label_generation_strategies.md` | How does the label-generation strategy affect degradation and the induction-circuit measures across generations? |
| `06_symbol_distribution_experiments.md` | What happens when the symbols themselves are fed back into the questions, or chosen more sharply? |

Finding 03 reported an earlier extended-task run made before that task had been
tuned; finding 05 supersedes it at the selected learning rate, so 03 was retired
and the number is not reused.

Each finding quotes the numbers it relies on, so it can be read without `results/`
in hand.

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
│   ├── base_task/
│   │   ├── train_original.py        train a model on the real task
│   │   ├── tune_learning_rate.py    the learning-rate search
│   │   ├── run_recursive.py         generate data from a parent, train successors
│   │   └── analyse_runs.py          per-run analysis and the cross-generation comparison
│   └── extended_task/
│       ├── extended_model.py        the backbone plus a two-way symbol head
│       ├── tune_extended.py         the extended task's learning-rate search
│       ├── run_extended.py          trains one extended chain per condition
│       └── analyse_extended.py      per-generation analysis and the comparisons
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
| `extended_model.py` | The extended task's model and losses: the authors' backbone unchanged, plus a two-way head that picks the next symbol. Also generates a continuation autoregressively. |
| `tune_extended.py` | Trains the extended task's learning-rate grid on correct continuations, applies the selection rule, writes the results table, selection record and figure. The recursive chains read the selected rate from that record. |
| `run_extended.py` | Trains one extended chain: generation 0 on correct continuations, then each successor on continuations its parent generated. `--label-strategy` and `--label-temperature` select the condition; generation 0 is trained once and shared by all of them. |
| `analyse_extended.py` | Per generation: development scores at all three output positions teacher-forced and self-generated, attention measures, single-head ablations. `--condition` writes one chain's table, figure and generated-data distributions; `--compare-family` puts the shared reference beside one family — label generation, context feedback or next-symbol temperature. |

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

### 6. Tune the extended task

```powershell
.\.venv\Scripts\python.exe scripts/extended_task/tune_extended.py
```

The same 6 × 3 grid as the base task, but trained on the **extended** task with
correct continuations. Selection uses final query-label accuracy on the
development set, averaged over seeds 5, 6 and 7; symbol loss and following-label
accuracy are recorded alongside but do not enter the rule.

**Selected: 1e-3**, again the best *among the tested rates at this budget*, again
at the top of the tested range. The grid was not expanded in response. Every
recursive command below reads this rate from
`results/extended_task/tuning/selection.json`, so the chains cannot drift from
the search that justified them.

All 18 candidates share one original-task training set, and finished candidates
are skipped, so the command is safe to stop and restart.

### 7. Run the nine recursive conditions

```powershell
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --generations 4
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --label-strategy sample --label-temperature 1 --generations 4
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --label-strategy sample --label-temperature 3 --generations 4
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --label-strategy sample --label-temperature 5 --generations 4
```

Those four are the **label-generation family**. Run them in order: the first
trains **generation 0** on correct continuations and then the argmax chain, and
the rest reuse that same generation 0. Generation 0 is trained once and shared by
every condition in every family.

The two symbol families change the questions and the symbol choice instead of the
labels, and both use argmax labels throughout:

```powershell
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --context-temperature 1 --generations 4
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --context-temperature 1/3 --generations 6
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --context-temperature 0.2 --generations 6
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --next-symbol-temperature 1/3 --generations 6
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --next-symbol-temperature 0.2 --generations 6
```

`--generations` names the **final** generation, not how many to add. Generations
that already exist are loaded as parents rather than retrained, so raising the
number extends a finished chain. **Chains deliberately differ in length**: the
four sharpened symbol conditions run to generation 6 because four generations
turned out not to be enough to characterise them, while the reference and context
temperature 1 stop at generation 4.

**Context feedback** (`--context-temperature`) changes where the question
generator gets its symbol weights. Before building each child's questions, the
parent is replayed over the openings in its *own* saved training set, the
identities of the next symbols it generates are counted, and those frequencies
`p` are sharpened into weights proportional to `p ** (1 / T)`. Temperature 1
leaves them as measured, 1/3 cubes them and 0.2 raises them to the fifth power.
The generator then builds ordinary valid questions from those weights — two
distinct context classes drawn without replacement, unchanged label pairs, query
construction and exemplar rules. Nothing is smoothed, replenished or cut off.
Because the two classes must differ, the frequencies actually offered need not
match the weights exactly; both are recorded.

**Next-symbol temperature** (`--next-symbol-temperature`) divides the two
next-symbol logits before sampling, so the parent follows its existing preference
between the two context symbols more sharply. That preference may be for a
position, for particular symbol identities, or both, and the analysis measures
those separately. Opening questions stay on the original distribution.

Both flags accept `1/3` so a third stays exact rather than rounded.

Finished generations are skipped, each condition writes to its own folder, and a
run trained at a different learning rate is refused rather than silently reused —
so the commands are safe to restart and the conditions cannot be mixed up.

**How a parent generates its successor's training data.** One token at a time,
each step conditioned on the tokens the parent itself produced:

1. the query label,
2. the next symbol, always **sampled** from its two-way distribution at
   temperature 1, in every condition,
3. the following label.

The two **labels** are where the conditions differ:

- **`label_argmax`** — each label is the single most likely one.
- **`label_sampling_temperature_1`**, **`_3`**, **`_5`** — each label is drawn
  from `softmax(logits / T)`. Dividing the logits by `T` flattens the
  distribution, so the higher the temperature the more often the parent writes
  down a label it did not think most likely. All five labels stay eligible at
  every temperature.

Mistakes are kept, and correct answers are never mixed back in. Opening contexts
and queries always come from the original task generator, in every condition.
Because generation is autoregressive, a sampled query label is fed back in before
the symbol is chosen, so the symbol probabilities differ between conditions even
though the symbol rule is identical.

Temperature applies **only** when generating a successor's training data: the
training loss, the evaluation decoding and generation 0's data are unchanged.

### 8. Analyse the extended task

Run `--condition` once per condition, then one `--compare-family` per family:

```powershell
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition label_argmax
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition label_sampling_temperature_1
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition label_sampling_temperature_3
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition label_sampling_temperature_5
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition context_feedback_temperature_1
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition context_feedback_temperature_one_third
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition context_feedback_temperature_0.2
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition symbol_sampling_temperature_one_third
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition symbol_sampling_temperature_0.2
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --compare-family label
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --compare-family context
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --compare-family symbol
```

Each family comparison puts the shared reference condition (`label_argmax`)
beside that family's conditions, so the three comparisons stay readable and
nothing is rerun to produce them.

Every condition is scored on the same fixed development questions from the
**original** task distribution — the evaluator is never adjusted to match a
condition's changing training distribution — with the same decoding and the same
checkpoint- and head-selection rules. A generation that has already been analysed
is reused rather than re-measured. Each per-condition
command also counts the distributions inside that condition's saved training
datasets — which symbols the parent chose, and where its labels went. The final
command reads the four tables and writes the comparison and the distribution
figure.

## What gets generated

`results/` is produced by the commands above and is not distributed with the
repository — regenerate it by following the steps in order.

| Path | Contents |
| --- | --- |
| `results/evaluation_data/` | `class_splits.json` (the evaluation protocol) and `eval_dev.h5` (the 1,000 fixed questions) |
| `results/base_task/tuning/` | one folder per tuning candidate, plus `results.json` (all 18 candidates), `selection.json` and `learning_rate_comparison.png` |
| `results/base_task/recursive/generation_<n>/` | one folder per generation |
| `results/base_task/recursive/generation_comparison.json` / `.png` | the across-generation comparison |
| `results/extended_task/tuning/` | one folder per tuning candidate, the shared original-task training set, plus `results.json` (all 18 candidates), `selection.json` and `learning_rate_comparison.png` |
| `results/extended_task/recursive/generation_0/` | the generation 0 shared by all four conditions |
| `results/extended_task/recursive/experiments/<condition>/generation_<n>/` | one folder per generation, per condition |
| `results/extended_task/recursive/experiments/<condition>/generation_comparison.json` / `.png` | that condition's chain |
| `results/extended_task/recursive/experiments/<condition>/dataset_distributions.json` | what that condition's parents actually generated, counted from the saved datasets: symbol offers, choices and selection rates per class, and for each label output the generated and true label histograms with a correct-versus-generated table |
| `results/extended_task/recursive/condition_comparison.json` / `.png` | the label family beside the reference |
| `results/extended_task/recursive/context_feedback_comparison.json` / `.png` | the context-feedback family beside the reference |
| `results/extended_task/recursive/symbol_sampling_comparison.json` / `.png` | the next-symbol-temperature family beside the reference |
| `results/extended_task/recursive/data_distributions.png` | where the generated labels went, label family |
| `results/extended_task/recursive/context_feedback_distributions.png` / `symbol_sampling_distributions.png` | symbol concentration, coverage, positional preference and identity preference for each symbol family |

Inside a generation folder:

| File | Contents |
| --- | --- |
| `config.json` | every resolved setting, including seeds and schedules. For an extended-task successor it also records the label strategy and temperature, the next-symbol temperature, the context-selection temperature, and — under context feedback — the parent's measured symbol frequencies and the sampling weights derived from them |
| `log.h5` | one training loss and gradient norm per update (31,250), plus development accuracy and loss at each evaluation point |
| `checkpoints/` | the saved models, named by sequence count |
| `generated_training_data.h5` | successors only: the training set as class and exemplar indices plus labels — compact, not copied feature vectors |
| `generation_metadata.json` | successors only: parent checkpoint, generation rule, how good the parent's answers were, and worked examples |
| `analysis/` | base task: `analysis.json` plus `curves.png`, `head_measures.png`, `attention_example.png` |
| `training_data.h5` | extended task: the opening contexts as indices plus the three continuation targets, and the correct answers kept for scoring |
| `dataset_quality.json` | extended task: how far the generated continuations departed from the correct ones (as counts **and** rates), which context position was chosen, and how the chosen symbol identities were spread |
| `analysis.json` | extended task: development scores at all three positions both teacher-forced and self-generated, attention measures per head, the selected heads and their ablation effects, and the measures across the analysed checkpoints |

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
only their final models are ever compared. That keeps each 18-model search small.
It also means a tuning candidate cannot support mechanistic analysis, which is
why generation 0 is trained separately rather than reused from the search — in
both tasks.

Analysis does not read all 55. It measures a fixed, roughly log-spaced subset of
**12** of them, by the same rule in every generation and every condition, so the
analysed points line up; each `analysis.json` lists exactly which ones it used.

Reducing snapshots never reduces the learning curves: those come from `log.h5`,
which is written at every evaluation point regardless.

## Limitations

- **Controlled seeds.** Every chain is a single recursive line whose models share
  an initialisation seed and training-data seed. That isolates the effect of the
  changing targets, but it is not a sample of independent chains.
- **Attention patterns are not causal claims.** A high induction score for a
  head says where it attends, not what it contributes to the output. Single-head
  zero-ablation measures the effect of that one intervention on one evaluator; a
  small effect does not establish redundancy, and no combination of heads was
  silenced.
- **Attention is sampled sparsely.** The attention measures and ablations are
  computed at 12 of the 55 checkpoints, while accuracy and loss are logged 201
  times per run. Nothing is claimed about what happened in between.
- **Degradation across generations and development within training are separate
  questions.** Where two measures move within the same observed interval, nothing
  here establishes which moved first.
- **Every successor starts from fresh weights.** No model inherits its parent's
  parameters, so what passes between generations is the training data, not a
  trained circuit.
- **Argmax versus sampling and temperature versus temperature are different
  comparisons.** Argmax against any sampling condition compares generation
  strategies; temperatures 1, 3 and 5 compare temperatures within sampling. An
  argmax-to-temperature-3 difference does not isolate temperature.
- **One chain per condition from a shared parent.** Four chains that share
  generation 0 are not four independent replications, and conclusions have to be
  read at that scope.
- **Both learning-rate searches stop at the edge.** 1e-3 was best of six rates at
  31,250 updates in each search, and sits at the top of both tested ranges.
- The reserved 100 final-test classes have never been scored.

## Attribution

The task, model, samplers, training loop, evaluation and the progress-measure
definitions are from Singh et al. (2024), used under the terms of their
repository. Ours is the evaluation protocol, the recursive pipeline, the tuning
search, the analysis and the write-ups in `findings/`.
