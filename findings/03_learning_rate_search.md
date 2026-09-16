# Finding 03 — The authors' learning rate is not the best one for this budget

15–16 September 2026 | source runs: `results/base_task/tuning/` and
`results/base_task/generation_0_original_data_init_seed_5`

## Question

Everything so far used the authors' learning rate, 1e-5. Before committing to a
long chain of recursive generations — where every generation must share one
fixed training setting — we need to know whether that rate is actually a good
choice **for our training budget** of 1,000,000 sequences (31,250 updates).

A second question arrived with it. We score models on a fixed set of
development questions drawn from 100 held-out character classes. Is 1,000
questions enough to choose between models, or does it need to be 10,000?

## Setup

A fixed grid of **five learning rates × three initialisation seeds = 15 models**:

- learning rates 1e-6, 3e-6, 1e-5, 3e-5, 1e-4
- initialisation seeds 5, 6, 7

The learning rate is the thing being tuned. The seeds are repeated starting
conditions: they tell us how much of any gap between rates is just noise.

Every candidate trains on the **original** task with **true** query targets — no
generated data anywhere in this stage. Architecture, optimizer (constant-rate
Adam), batch size 32, and the training-data seed are identical across all 15, so
every candidate sees the same 1,000,000 questions in the same order. Each trains
for exactly one pass, and only the **final** checkpoint is compared: no early
stopping and no best-checkpoint picking.

The candidate at (1e-5, seed 5) is the existing baseline run, reused rather than
retrained after checking its configuration matched.

**Selection rule, fixed before any result was inspected:** highest mean accuracy
over the three seeds, compared unrounded; an exact tie is broken by the lowest
mean loss; a remaining tie prefers the authors' 1e-5, otherwise the smaller rate.
A rate missing any seed is excluded entirely rather than averaged over a subset.

The two evaluators draw from the same 100 held-out classes under the same task
rules, differing only in size and random seed. The 10,000-question scores come
from the original search. The 1,000-question scores were then computed **fresh**,
by loading each of the 15 saved final checkpoints and running it on the
1,000-question set — not read out of any training log or earlier record.

## Results

![learning rate comparison](../results/base_task/tuning/learning_rate_comparison.png)

Mean final development accuracy across seeds 5, 6 and 7:

| learning rate | 1,000 questions | 10,000 questions | mean loss (1,000q) | mean loss (10,000q) |
| --- | ---: | ---: | ---: | ---: |
| 1e-6 | 0.5157 | 0.5197 | 0.7593 | 0.7571 |
| 3e-6 | 0.8127 | 0.8161 | 0.4135 | 0.4189 |
| 1e-5 (authors') | 0.9677 | 0.9611 | 0.0928 | 0.1142 |
| 3e-5 | 0.9790 | 0.9766 | 0.0822 | 0.1037 |
| **1e-4** | **0.9923** | **0.9911** | **0.0278** | **0.0349** |

Per-seed accuracy on the 10,000-question set:

| learning rate | seed 5 | seed 6 | seed 7 | spread |
| --- | ---: | ---: | ---: | ---: |
| 1e-6 | 0.5224 | 0.5184 | 0.5184 | 0.002 |
| 3e-6 | 0.8383 | 0.7656 | 0.8443 | 0.036 |
| 1e-5 | 0.9559 | 0.9686 | 0.9587 | 0.005 |
| 3e-5 | 0.9713 | 0.9818 | 0.9766 | 0.004 |
| 1e-4 | 0.9889 | 0.9918 | 0.9927 | 0.002 |

**Selected learning rate: 1e-4, under both evaluators.**

The ranking is identical under both: 1e-6 < 3e-6 < 1e-5 < 3e-5 < 1e-4, with no
crossings. Every gap between adjacent rates is much larger than the spread
across seeds, except between 1e-5 and 3e-5 where the gap (about 1.1 points) is
still roughly twice the seed spread.

Per-candidate, the two evaluators agree to within 1.1 accuracy points on 14 of
15 models. The exception is (1e-6, seed 6), which scores 0.495 on the small set
against 0.518 on the large one — a 2.3-point disagreement, at the learning rate
whose models sit closest to chance and are therefore least stable.

The reserved final-test classes were not generated and not scored.

## Interpretation

**The authors' 1e-5 is not the best rate at this budget — 1e-4 beats it by
about 3 accuracy points and roughly a third of the loss.** This is not a
contradiction of their work: they were not tuning for our budget, and a slower
rate that has not converged after 31,250 updates may well be the better choice
over a longer run. What it does mean is that continuing with 1e-5 would have
handicapped every generation of the recursive experiment.

The 1e-6 result is **genuine under-training, not divergence**: losses stayed
finite and were still falling at the end of training, and accuracy was stuck
near 50% — the chance level once predictions are restricted to the two labels
present in the context. At that rate the model learns to bet on the two context
labels but never learns which of them is right within the budget.

**On the evaluator question, the two sets agree.** They select the same rate,
produce the same ranking, and differ by about a point per model. For a decision
this clear-cut, 1,000 questions would have been enough. The 10,000-question set
still measures more precisely — its per-rate seed spread is smaller at the top
of the range, and it would matter for a closer comparison, such as generations
whose accuracies differ by a fraction of a point.

**The win is at the edge of what we tested.** 1e-4 is the largest rate in the
grid, so the true optimum may lie above it. The correct description of this
result is *best among the tested rates under this training budget* — not "the
best learning rate". The grid was deliberately not expanded.

## Limitations and unresolved

- **Boundary result.** 1e-4 is the top of the range. 3e-4 and 1e-3 were not run.
  Nothing here rules out a better rate above 1e-4, or instability there.
- **One budget.** Every conclusion is specific to 31,250 updates. The ordering
  could change over a longer run, which is exactly why 1e-6 looks so bad.
- **Three seeds.** Enough to see that the gaps exceed seed noise, not enough for
  a confidence interval.
- **Final checkpoints only.** No candidate was examined mid-training, so we
  cannot say whether the higher rates reach their accuracy sooner or simply end
  higher.
- **No mechanistic analysis of the tuned models.** Whether an induction circuit
  forms in the same way at 1e-4 as at 1e-5 is untested. That matters for the
  project's central question and should be checked before the recursive chain is
  read mechanistically.
- **Pending decision:** which development evaluator to keep permanently. Both
  remain in the repository until the user decides.

## Reproduce

From the project root, with the environment set up (see README):

```powershell
.\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py --compare-evaluators
```

The first tuning command trains any candidate that does not already have a
result and writes the selection; the second re-scores the 15 saved final
checkpoints on the 1,000-question set and writes the comparison and the figure.
Neither retrains a finished candidate.

Measurements and figure:

- `results/base_task/tuning/results.json` — the grid record and its
  10,000-question scores
- `results/base_task/tuning/selection.json` — the selection under the
  10,000-question set, and the chosen generation-0 checkpoint
- `results/base_task/tuning/evaluator_comparison.json` — per-candidate scores on
  both evaluators, per-rate means, and the winner under each
- `results/base_task/tuning/learning_rate_comparison.png` — the figure above

**The model this selects as generation 0 for the recursive experiment:**
`results/base_task/tuning/learning_rate_0.0001_init_seed_5/checkpoints/00001000000.eqx`

Seed 5 was fixed in advance as the parent seed. It is not the best-scoring seed
of the three — seed 7 scored slightly higher — and was deliberately not chosen
on performance.
