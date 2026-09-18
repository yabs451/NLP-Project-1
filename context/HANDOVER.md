# Project handover

Everything a new teammate (or their AI assistant) needs to understand this
project and continue it, without access to the conversations that produced it.
Written 18 September 2026.

Numbers here are quoted from the files under `results/`; each section says which.
Where something is unverified or unresolved, it says so.

---

## 1. What the project is about

We study **recursive training**: train a model, have it generate synthetic
training data, train a fresh model on that data, and repeat. The question is what
degrades, in what order, and whether anything about the model's internal
machinery changes before its predictions do.

The task is **in-context learning**. Each question carries its own
symbol-to-label mapping, so the answer must be read out of the context rather
than memorised. The mechanism that solves this is usually described as an
**induction circuit**: a *previous-token head* in layer 0 lets each label token
carry information about the symbol before it, and an *induction head* in layer 1
lets the query attend to the label that followed the matching symbol.

Everything is built on **Singh et al. (2024), "What needs to go right for an
induction head?"** Their code is vendored unmodified in `upstream/icl-dynamics/`
(commit `85b895c720844e795734b9391b32d2619065f9a1`). We use their model,
samplers, optimizer, loss, training loop, evaluation and checkpointing, and add
our own evaluation protocol, recursive pipeline, tuning searches and analysis.

### The original question, and its current status

> Does induction-circuit function weaken *before* overall predictive accuracy
> declines?

**This has not been established.** Where both change, they change within the same
observed generation interval, and generations are the only time resolution we
have across a chain. One partial dissociation recurs: substantial *loss*
increases show up at generations where *accuracy* is still high. Another runs the
opposite way (section 5). Treat the ordering as open.

---

## 2. The two tasks

### Base task

Five tokens: two symbol–label pairs, then a query symbol matching one of them.
The model predicts the query's label, one of five.

```
symbol A → label 3 , symbol B → label 1 , query = symbol A   ⟹  answer 3
```

Labels are reassigned at random for every question, so no fixed symbol-to-label
mapping helps. Symbols are precomputed ResNet18 feature vectors of Omniglot
characters — a symbol is a 512-number vector, not an image the model sees.

### Extended task

Same opening, then **three** predictions in order:

1. the **query label** (as above);
2. a **next symbol**, chosen from the two symbols already in the context;
3. **that symbol's label**.

```
symbol A → 1 , symbol B → 4 , query = A   ⟹   1 ,  then B ,  then 4
```

Both label predictions have exactly one right answer. **Either** symbol choice is
legitimate, so the symbol target is a fair coin flip: the best a model can do is
learn that distribution, which puts its cross-entropy at ln 2 ≈ 0.693. Note that
cross-entropy is *not* bounded near ln 2 — a confidently wrong symbol head scores
far above it, and one of our conditions reached 5.75.

Restricting the next symbol to the two context symbols is **our design choice**,
not part of the original task.

The only new parameters are a two-way `Linear(64 → 2)` head reading the hidden
state at the next-symbol position. The backbone's own label head is reused for
both label predictions. Rotary positions mean going from five to seven tokens
needs no architectural change.

---

## 3. What comes from where, in each experiment

This is the distinction that matters most and is easiest to lose.

| | opening question (symbols, context labels, query) | query label | next symbol | following label |
| --- | --- | --- | --- | --- |
| Base recursive chain | original generator | **parent** (argmax) | — | — |
| Extended, label family | original generator | **parent** | **parent** (T = 1) | **parent** |
| Extended, context feedback | generator, but with **class weights derived from the parent's generated symbols** | **parent** (argmax) | **parent** (T = 1) | **parent** (argmax) |
| Extended, next-symbol temperature | original generator | **parent** (argmax) | **parent** (T = 1/3 or 0.2) | **parent** (argmax) |

In every condition the question generator still builds valid questions with the
original rules: two distinct context classes, the original label-pair and query
construction, the original exemplar rules. Context feedback changes only *which
symbol identities the generator is more likely to pick*, never the rules.

### Shared generation 0, and fresh weights

Generation 0 is trained once on correct data and **shared by every extended-task
condition**. Conditions differ only in how their successors' data is made.

Every successor starts from **fresh weights and a fresh optimizer** (initialisation
seed 5). A child never inherits its parent's parameters. This matters for
interpretation: nothing about a parent's internal structure is passed along
physically. What passes between generations is *the training data*. So it is wrong
to describe a chain as a circuit being progressively eroded — the right question
is whether each separately trained child *develops* the pattern during its own
training.

---

## 4. Settings, and what they were chosen by

| | value |
| --- | --- |
| Learning rate | **0.001** for both tasks (see below) |
| Initialisation seed | 5 for every recursive model |
| Batch size | 32 |
| Training budget | 1,000,000 examples = 31,250 updates, one pass |
| Checkpoints, recursive runs | **55 per run**, written directly during training — the initialisation, four early snapshots near 1,000 / 2,000 / 5,000 / 10,000 sequences (landing at 1,024 / 2,016 / 5,024 / 10,016), then every 20,000 to 1,000,000. That includes both endpoints, so 53 are intermediate |
| Checkpoints, tuning candidates | **2 per run** — initialisation and final only, because only final models are compared. A tuning candidate therefore cannot support mechanistic analysis, which is why generation 0 is trained separately |
| Development evaluator | a fixed **1,000-question** set from 100 held-out classes, never changed between conditions |
| Accuracy/loss logging | every 5,000 sequences → **201 points** per run, in `log.h5` |
| Attention + ablation analysis | **12 of the 55 checkpoints** in the extended-task analyses, the same 12 in every extended run. The base-task analysis uses its own rule and reports **13** analysed checkpoints, so the two tasks are not directly comparable on this axis |
| Final test | **100 reserved classes, never generated and never scored.** Outstanding by design: it should be run once the experimental choices are settled, so the held-out number is reported against a final configuration rather than a moving one |

### The two learning-rate searches

Both tasks got their own search: six rates (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3)
× three **backbone** initialisation seeds (5, 6, 7) = 18 candidates each, trained
on correct data and compared on final checkpoints.

- Base task: selected **0.001**, mean accuracy 0.9973 (`results/base_task/tuning/selection.json`)
- Extended task: selected **0.001**, mean query accuracy 0.9977 against 0.9570 at
  the authors' 1e-5 (`results/extended_task/tuning/selection.json`)

Both are **best among the tested settings at this budget**, and both winners sit
at the **top edge of the tested range** — recorded as `winner_at_grid_boundary:
true`.

The base search got there in two stages: it began with five rates (1e-6 to 1e-4),
1e-4 won at the boundary, and 1e-3 was then added, after which 1e-3 won. The
extended search ran all six rates at once. **Neither search tested anything above
1e-3**, so nothing here rules out a better — or an unstable — rate further up.

Only the backbone weights vary with the seed; the two-way symbol head keeps its
own fixed seed (11) and the training data is identical across candidates.

**Seed 5 was fixed in advance** as the seed carried into the recursive chains. It
is not the best-scoring seed — seeds 6 and 7 both scored slightly higher.

Seeds used, recorded in every run's `config.json`: symbol-head initialisation 11,
generation-0 coin flips 12, development coin flips 13, next-symbol sampling 14,
label sampling 15, context-feedback frequency pass 16.

### Two pieces of history

- **The evaluator changed once, early.** A 10,000-question development set was
  used for the first five base-task rates, then all 15 models were rescored on
  the 1,000-question set. The two agreed on the ranking and the winner, so the
  smaller set was adopted everywhere and the larger one retired. The 1e-3
  candidates were never scored on 10,000 questions.
- **An earlier extended-task experiment ran at the authors' 1e-5**, before that
  task had been tuned. Once the search selected 1e-3 it was superseded and moved,
  with its finding, out of the maintained project.

Both retired items live in `temporary_checks/`, which is **local and untracked**
and is **not a dependency**: no script imports it, no command names it, no active
finding cites it. A teammate will not receive it and does not need it.

---

## 5. Experiment inventory and headline results

All extended conditions live in
`results/extended_task/recursive/experiments/<condition>/` and share
`results/extended_task/recursive/generation_0/`.

### Base task — `results/base_task/recursive/`

Generations 0–4 at 0.001. **The chain preserved every quantity we measured**:
development accuracy 0.9960 and loss 0.0114 at all five generations, induction
0.9596, previous-token 0.9999, identical head selections. The parent's answers
matched the true answers on all 1,000,000 examples at every generation, so each
successor effectively retrained on the original task. Final parameters were equal
wherever compared.

*Reading*: a null result, and an informative one — with uncorrupted targets,
recursion changed nothing measurable here.

### Extended, label family — reference for both symbol families

| condition | generations | query accuracy, gen 0 → last |
| --- | --- | --- |
| `label_argmax` ← **shared reference** | 0–4 | 0.997 → 0.993 |
| `label_sampling_temperature_1` | 0–4 | 0.997 → 0.999 |
| `label_sampling_temperature_3` | 0–4 | 0.997 → **0.489** |
| `label_sampling_temperature_5` | 0–4 | 0.997 → **0.194** |

Argmax and temperature 1 stayed broadly stable. Temperatures 3 and 5 deteriorated
substantially on both label outputs, temperature 5 earlier in this comparison
(between generations 1 and 2, against 2 and 3 for temperature 3). Generated label
frequencies moved towards approximately even across the five labels, and symbol
identities stayed broadly balanced. Symbol loss stayed near ln 2 throughout — the
deterioration was specific to the labels, whose targets were the ones corrupted.

Full write-up: `findings/05_label_generation_strategies.md`.

### Extended, context feedback — reference is `label_argmax`

Each child's questions are built from class weights `w ∝ p ** (1/T)`, where `p`
comes from a **separate frequency-estimation pass**, not from any saved dataset's
stored choices. Before each child is built, the parent is re-run over the opening
questions recorded in **its own** training dataset, generating fresh continuations
(argmax labels, next symbol at the condition's temperature), and the identities of
the next symbols it produces in that pass are counted. The counts are accumulated
in memory; no extra million-example file is written. Generation 1's weights come
from replaying generation 0 over generation 0's questions, and so on.

| condition | generations | query accuracy |
| --- | --- | --- |
| `context_feedback_temperature_1` | 0–4 | 0.997 → 0.992 (little change) |
| `context_feedback_temperature_one_third` | 0–6 | 0.997 at g4 → 0.974 at g5 → **0.410** at g6 |
| `context_feedback_temperature_0.2` | 0–6 | 0.994 at g3 → **0.535** at g4 → 0.615 at g5 → 0.526 at g6 |

Distinct symbol classes selected in each million-example dataset:
temperature 1/3 goes 50 → 50 → 34; temperature 0.2 goes 50 → 40 → 12 → **5**.

**Temperature 1/3 fell to 0.410 with exactly zero recorded label errors in all six
of its datasets** — both label outputs, every generation. For that chain,
corrupted targets are ruled out as an explanation: there were none.

Temperature 0.2 is not in that position and should not be described as if it
were. Its datasets recorded 0 query / 0 following errors at generations 1, 2, 4
and 5; **19 query and 25 following** at generation 3; and **5 query and 8
following** at generation 6. Those counts are small against 1,000,000 examples,
but small is not zero, and we have not shown that a few hundred wrong targets
have no effect — we simply have no condition isolating that.

In both chains the questions changed substantially at the same time. That the
concentration *caused* the fall is not established by these runs.

Temperature 0.2 did not keep falling; it fluctuated between 0.52 and 0.62 after
generation 4. Temperature 1/3 looked *stable* at generation 4 and had fallen
sharply by generation 6 — which is the clearest reason in this project to distrust
"stable" claims made at a fixed number of generations.

Full write-up: `findings/06_symbol_distribution_experiments.md`.

### Extended, next-symbol temperature — reference is `label_argmax`

Two different things are being measured here, over different index ranges, so the
columns are labelled explicitly. **Position share and coverage describe generated
datasets**, and the dataset for generation *g* was written by generation *g−1*;
**symbol loss, query accuracy and following accuracy describe trained models**,
where generation 0 is the shared parent.

| condition | position-0 share, dataset g1 → g6 | symbol loss, model g0 → g6 | query acc, model g6 | following acc, model g6 |
| --- | --- | --- | --- | --- |
| `symbol_sampling_temperature_one_third` | 0.462 → 0.116 | 0.694 → 4.13 | 0.993 | 0.961 |
| `symbol_sampling_temperature_0.2` | 0.438 → 0.120 | 0.694 → 5.75 | 0.997 | 0.940 |

Colder sampling produced mainly a **context-position** preference, which plateaued
near 0.12 rather than going to zero. Symbol-identity concentration barely moved:
all 50 classes selected in every generation, entropy 3.9118–3.9120 against the
reference's 3.9120. The symbol output degraded steadily across all six
generations.

**Following-label accuracy dipped and then partly recovered** in the
temperature-0.2 chain: 0.971 at generation 3, **0.898** at generation 4, then
0.935 and 0.940. That dip is the clearest sign that "the label outputs were
unaffected" would be the wrong summary.

**Query-label accuracy did not rise above its starting point.** In the
temperature-0.2 chain it runs 0.997 (g0), 0.995, 0.995, 0.991, 0.991, 0.989,
0.997 (g6) — a shallow dip across the middle generations and a return to exactly
generation 0's value, not a net gain.

**A dissociation worth knowing about**: along that same chain the strongest
induction score fell 0.977 → 0.646 while query accuracy recovered to its original
0.997. Two measurements moving apart is all that establishes. It is not evidence
that circuit function disappeared, and not evidence that it survived.

---

## 6. Distinctions a new reader will otherwise miss

1. **Labels vs symbol identities vs context positions.** A *label* is one of five
   class tokens. A *symbol identity* is which of the 50 Omniglot classes appears.
   A *context position* is whether the model copied slot 0 or slot 1. Three
   different things; our conditions move different subsets of them.
2. **Sampling weights vs realised frequencies.** Under context feedback a class
   reached 0.9967 of the *sampling weight*, but appeared in 0.5000 of offered
   context slots and 0.5001 of generated symbols. Because the two context classes
   must differ, a class can fill at most one of two slots, so its **offered** share
   genuinely cannot exceed 0.5. Its **selected** share is *not* capped — a parent
   that always picked the dominant slot would generate it far more often. The
   near-0.5 selected share is a measurement, not arithmetic.
3. **Sampled output frequencies vs model probability vectors.** Everything in
   `dataset_distributions.json` counts tokens that were actually drawn, and many
   different per-question probability distributions produce the same marginal. So
   **that analysis** does not recover per-question probabilities. Other parts of
   the project do use probabilities — cross-entropy loss is computed from them
   throughout, and finding 01 reports a direct diagnostic measuring how much
   probability mass sat on the two context labels. The limitation is specific to
   the generated-data distribution counts, not a claim that probabilities were
   never examined.
4. **Accuracy vs loss vs attention score vs ablation effect.** These move
   independently and repeatedly disagree. Loss rose sharply while accuracy held;
   induction fell while accuracy rose. Ablation effects depend strongly on *which*
   head is silenced: removing the strongest **induction** head changed query
   accuracy by at most about 3.5 percentage points anywhere in the project, but
   removing the strongest **previous-token** head reached about **41 percentage
   points** (context feedback at temperature 0.2, generation 1) and about 29
   points in the label-sampling temperature-1 chain. Do not treat any one of these
   measures as a proxy for another, and do not generalise "ablation effects are
   small" across head types.
5. **The dataset for generation *g* was written by generation *g−1*.** "Dataset
   for generation 4" is generation 3's output, used to train generation 4.
6. **1,000 vs 1,000,000.** Development measurements are on 1,000 fixed questions.
   Dataset corruption counts are over 1,000,000 training examples. Never compare
   the two directly.
7. **An empirical zero becoming a permanent zero.** The zero that matters is a
   zero **in the frequency-estimation pass**: if a class is never selected during
   that one pass over a million questions, its measured frequency is exactly 0.
   Our rule applies `p ** (1/T)` with no smoothing and no floor, so the weight is
   exactly 0, the class is never offered in the child's questions, cannot be
   selected in the next pass either, and is recorded as 0 from then on. That is
   our implementation acting on one finite sample — not a measurement that the
   model assigns the class zero probability. A small non-zero probability can
   easily produce no draws in a million samples.
8. **Sharpening is an intervention.** The exponent in the context-feedback rule is
   something we imposed. The resulting concentration is not spontaneous model
   behaviour.

---

## 7. How to run things

Setup (Python 3.10, CPU only — JAX has no native-Windows GPU build):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then, from the project root, in this order:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py
.\.venv\Scripts\python.exe scripts/base_task/train_original.py --results-subfolder recursive --run-name generation_0 --learning-rate 0.001 --init-seed 5
.\.venv\Scripts\python.exe scripts/base_task/run_recursive.py --parent results/base_task/recursive/generation_0 --generations 4
.\.venv\Scripts\python.exe scripts/extended_task/tune_extended.py
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --generations 4
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --context-temperature 1/3 --generations 6
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition label_argmax
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --compare-family context
```

**The base chain needs its generation 0 trained first** — `run_recursive.py` only
trains successors and requires an existing `--parent`. Note that
`train_original.py` defaults to the authors' 1e-5, so `--learning-rate 0.001` must
be given explicitly; `--init-seed 5` is likewise explicit. Its `--save-checkpoints`
already defaults to `mechanistic` (the 55-checkpoint schedule).

The extended chain's generation 0 is different: `run_extended.py` trains it
automatically on its first run and reads the learning rate from
`results/extended_task/tuning/selection.json`, so it takes no rate argument.

The README lists every condition's command. `--label-strategy`,
`--label-temperature`, `--next-symbol-temperature` and `--context-temperature`
select the condition; temperatures accept `1/3` so a third stays exact.

### `--generations` means DIFFERENT things in the two scripts

This is the argument most likely to waste hours. **The two scripts do not share a
convention.**

**`run_extended.py` — the FINAL generation number.** It loops `1 .. N`; any
generation that already has a `log.h5` is loaded as the parent, not retrained;
only missing ones are trained.

- `--generations 6` on a chain ending at 4 trains exactly generations 5 and 6.
- `--generations 4` on that same chain trains nothing and exits.
- It refuses to build on a run trained at a different learning rate.

**`run_recursive.py` (base task) — how many ADDITIONAL successors to train.** It
loops `for _ in range(N)`, each time numbering the next folder from its parent.

- `--parent .../generation_4 --generations 2` trains generations 5 and 6.
- It does **not** skip completed work. If the target folder already exists it
  **stops with an error** rather than resuming: *"generation_N already exists;
  refusing to overwrite"*. To continue a chain, point `--parent` at the last
  completed generation and ask only for the number still missing.

### What the analysis commands actually do

They are **not** all skip-if-done:

- `analyse_extended.py --condition X` reuses a generation's existing
  `analysis.json` when it already has the fields this version reports, so models
  are not re-measured. It **always** recounts the generated-data distributions and
  **always** rewrites that condition's comparison table and figures.
- `analyse_extended.py --compare-family F` always rewrites that family's
  comparison outputs. It only reads tables the per-condition runs wrote.
- `analyse_runs.py <run folder>` (base task) **always recomputes** that run's
  analysis — there is no skip check.
- `analyse_runs.py --compare <folder>` only reads the per-run analyses, so every
  generation must already have been analysed.

Re-running an analysis is therefore safe but not free, and it overwrites the
previous outputs for that condition.

### Where the outputs are

- `results/<task>/tuning/` — the learning-rate grids, `results.json`,
  `selection.json`, figure
- `results/extended_task/recursive/generation_0/` — the shared parent
- `.../experiments/<condition>/generation_<n>/` — `config.json` (every setting and
  seed, plus the context-feedback weights), `log.h5` (201-point curves),
  `checkpoints/` (55), `training_data.h5`, `dataset_quality.json`, `analysis.json`
- `.../experiments/<condition>/generation_comparison.*` and
  `dataset_distributions.json` — one chain
- `.../recursive/condition_comparison.*`, `context_feedback_comparison.*`,
  `symbol_sampling_comparison.*` and the three `*_distributions.png` — family
  comparisons

`findings/` holds the written science. It quotes the numbers it relies on inline,
so the argument can be followed as text — but it is **not** self-contained:
every finding embeds figures from `results/` and cites the files there for the
per-class arrays, full tables, learning curves and checkpoint trajectories it does
not reproduce. `results/` is part of the handover, not an optional extra.

---

## 8. What is done, what is not, and what is unresolved

**Complete**: both tuning searches; the base chain (0–4); nine extended
conditions; all analyses and figures; findings 01, 02, 04, 05, 06.

**Outstanding by design**: the **reserved 100 final-test classes have never been
generated or scored**. The intention has always been to run this *after* the
experimental choices are finalised, so that the held-out number describes a
settled configuration. It is the obvious next deliverable once the project stops
adding conditions.

**Genuinely unresolved**, recorded rather than explained:

1. Why silencing the strongest previous-token head costs under 1 point in some
   runs and up to 41 points in others.
2. Why the temperature-0.2 context chain partially recovered at generation 5.
3. Why the next-symbol-temperature chains reach 90% accuracy *earlier* each
   generation.
4. Whether the induction pattern appeared at any of the 43 checkpoints per run we
   never analysed.
5. Whether any collapsed run would recover with a longer budget. Untested, and the
   data do not indicate either way.

**Standing limitations.** One chain per condition, one initialisation seed, one
budget — these are not replications. Attention measures come from 12 of 55
checkpoints. Development numbers are a 1,000-question sample, not the whole
question distribution. Both tuning winners sit at the edge of their grids. For
context feedback, training and evaluation distributions diverge by design: those
scores describe the *original* question distribution, and we never scored those
models on the distribution they were trained on.

**Proposals, not agreed tasks.** Repeat one condition at seeds 6 and 7 to see
whether the collapse generation is stable. Score a context-feedback child on its
own training distribution to separate distribution shift from capability loss.
Ablate heads in combination rather than singly. Analyse more of the 55 checkpoints
around a transition. Score the reserved final test once the project is ready.

---

## 9. Working conventions

These are the rules the project has been run under. A teammate's AI assistant
should follow them.

- **Minimality first.** Before adding any file, option or output, answer: which
  current experiment, result, figure or reproduction step needs it? If there is no
  answer, do not add it. Never justify something by future usefulness.
- **One implementation per experiment**, not a script per variant. All nine
  extended conditions run through `run_extended.py`.
- **Leave `upstream/` alone.** Reuse the authors' functions; write a small adapter
  in `scripts/` if something cannot be done through their options, and say so.
- **Never fabricate a measurement.** Adapt the analysis to the checkpoints that
  exist.
- **Scientific writing**: separate what was observed from what explains it from
  what is unestablished. Attention patterns are not causal claims. State what was
  actually checked. Report null results honestly. Correct earlier documents openly
  and say whether any number changed.
- **On changing settings.** Running a *new* experiment at different settings is
  legitimate and encouraged where there is a scientific reason — that is how the
  temperature families and the extension to generation 6 came about. What is not
  acceptable is silently altering an experiment that has already been reported,
  or presenting an exploratory choice as though it had been planned in advance.
  Document the reason, keep the earlier result, and say which is which.
- **Two document types**: `findings/` is the scientific record (question, setup,
  results, interpretation, limitations, reproduction); `Development/reports/` is
  operational history, numbered by stage and dated. Do not rewrite a dated report
  to match later knowledge — correct errors in place and mark superseded material.
- **Do not copy measurements into `findings/`.** A finding quotes the numbers it
  needs and points at the file under `results/`.
- **Generations**: generation 0 is the original model; "N additional generations"
  means N successors, so the total is N+1.
- Keep operational matters — environment problems, scheduling, debugging — out of
  `findings/`. They belong in `Development/reports/`.
