# Finding 03 — The authors' learning rate is far from the best one for this budget

15–16 September 2026 | source runs: `results/base_task/tuning/`

## Question

Everything before this used the authors' learning rate, 1e-5. Before committing
to a chain of recursive generations — where every generation must share one
fixed training setting — we needed to know whether that rate is a good choice
**for our training budget** of 1,000,000 sequences (31,250 updates).

## Setup

A grid of **six learning rates × three initialisation seeds = 18 models**:

- learning rates 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3
- initialisation seeds 5, 6, 7

The learning rate is what is being tuned. The seeds are repeated starting
conditions: they show how much of a gap between rates is just noise.

Every candidate trains on the **original** task with **true** query targets — no
generated data anywhere in this stage. Architecture, optimizer (constant-rate
Adam), batch size 32 and the training-data seed are identical across all 18, so
every candidate sees the same 1,000,000 questions in the same order. Each trains
for exactly one pass from fresh weights and a fresh optimizer, and only the
**final** checkpoint is scored: no early stopping, no best-checkpoint picking.

All 18 are scored on the fixed 1,000-question development evaluator, built from
100 character classes that no training run ever sees.

**Selection rule, fixed before any result was inspected:** highest mean accuracy
over the three seeds, compared unrounded; an exact tie broken by lowest mean
loss; a remaining tie preferring the authors' 1e-5, otherwise the smaller rate. A
rate missing any seed is excluded rather than averaged over a subset.

**The search was extended once, and honestly.** The first five rates (1e-6 to
1e-4) were run as one grid, and the winner came out at the top of that range.
1e-3 was therefore added afterwards — it was not planned from the start. The
search then stopped: 3e-4 and everything above 1e-3 remain untested.

## Results

![learning rate comparison](../results/base_task/tuning/learning_rate_comparison.png)

Mean final accuracy on the 1,000 development questions, across seeds 5, 6, 7:

| learning rate | mean accuracy | spread (std) | mean loss | seed 5 | seed 6 | seed 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-6 | 0.5157 | 0.0150 | 0.7593 | 0.530 | 0.495 | 0.522 |
| 3e-6 | 0.8127 | 0.0349 | 0.4135 | 0.844 | 0.764 | 0.830 |
| 1e-5 (authors') | 0.9677 | 0.0033 | 0.0928 | 0.967 | 0.972 | 0.964 |
| 3e-5 | 0.9790 | 0.0036 | 0.0822 | 0.974 | 0.981 | 0.982 |
| 1e-4 | 0.9923 | 0.0005 | 0.0278 | 0.992 | 0.992 | 0.993 |
| **1e-3** | **0.9973** | 0.0009 | **0.0099** | 0.996 | 0.998 | 0.998 |

**Selected learning rate: 1e-3.** All 18 candidates trained to completion; none
diverged, and every loss stayed finite.

Accuracy rises monotonically across the whole range, but the curve flattens
hard above 1e-5: the step from 1e-5 to 1e-3 is 3.0 accuracy points, while the
step from 1e-6 to 1e-5 is 45 points. Loss keeps falling faster than accuracy
suggests — 0.0928 at 1e-5 against 0.0099 at 1e-3, nearly a tenfold reduction —
so the higher rates are not merely getting more answers right, they are much
more confident about them.

Seed spread shrinks as the rate rises: ±0.035 at 3e-6 against ±0.001 at 1e-3.
Every gap between adjacent rates is larger than the seed spread at that rate.

The reserved final-test classes were not generated and not scored.

## Interpretation

**The authors' 1e-5 is a poor choice at this budget.** 1e-3 is a hundred times
larger and reaches 99.7% against 96.8%, with an order of magnitude lower loss.
This does not contradict their work: they were not tuning for a 31,250-update
budget, and a slower rate that has not converged here might be the better choice
over a much longer run. What it does mean is that continuing at 1e-5 would have
handicapped every generation of the recursive experiment.

The 1e-6 result is **genuine under-training, not divergence**: losses stayed
finite and were still falling at the end, and accuracy sat near 50% — the chance
level once predictions are restricted to the two labels present in context. At
that rate the model learns to bet on the two context labels but never learns
which is right within the budget.

**The win is again at the edge of what was tested**, and this time we know the
curve is flattening rather than still climbing steeply. The correct description
remains *best among the tested rates under this training budget* — not "the best
learning rate". Nothing here rules out a better rate above 1e-3, or instability
somewhere above it.

## Historical note: the evaluator-size comparison

Earlier the project maintained a second, 10,000-question development evaluator
and used it for the original five-rate search. The 15 candidates from that search
were then re-scored on the 1,000-question set to check whether the smaller set
was precise enough. It was: both evaluators produced the same ranking and
selected the same rate, agreeing within 1.1 accuracy points on 14 of 15 models.

The 10,000-question evaluator was therefore retired, and the project now uses
the 1,000-question set throughout. **The 1e-3 candidates were never scored on
10,000 questions**, so that comparison covers the original 15 models only.

Measurements: `results/base_task/tuning/evaluator_comparison.json` and
`results/base_task/tuning/evaluator_size_comparison.png`. The retired evaluator
can be rebuilt from `temporary_checks/rebuild_large_dev_evaluator.py`.

## Limitations and unresolved

- **Boundary result.** 1e-3 is the top of the range and the search stopped there
  by decision. 3e-4 and rates above 1e-3 were not tested.
- **One budget.** Every conclusion is specific to 31,250 updates. The ordering
  could change over a longer run, which is exactly why 1e-6 looks so bad here.
- **Three seeds.** Enough to show the gaps exceed seed noise, not enough for a
  confidence interval.
- **Final checkpoints only.** No candidate was examined mid-training, so we
  cannot say whether the faster rates reach their accuracy sooner or simply end
  higher.
- **The selected model has no intermediate checkpoints.** Tuning runs save only
  the first and last, so the mechanistic analysis in findings 01 and 02 — which
  needs snapshots across training — cannot be run on it as it stands. Producing
  those measurements would require retraining at 1e-3 under the 55-checkpoint
  policy. That was deliberately not done here.
- **No mechanistic analysis at 1e-3.** Whether an induction circuit forms the
  same way at a hundred times the learning rate is untested, and it matters for
  the project's central question.

## Reproduce

From the project root, with the environment set up (see README):

```powershell
.\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py
.\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py
```

The tuning command trains any candidate without a result, scores each final
checkpoint on the 1,000-question set, applies the selection rule and writes the
figure. Finished candidates are skipped, so it is safe to stop and restart.

Measurements and figure:

- `results/base_task/tuning/results.json` — all 18 candidates: learning rate,
  seed, run path, accuracy and loss
- `results/base_task/tuning/selection.json` — the selection rule, per-rate
  summaries, the winning rate and the chosen checkpoint
- `results/base_task/tuning/learning_rate_comparison.png` — the figure above

**The model this selects as generation 0 for the recursive experiment:**
`results/base_task/tuning/learning_rate_0.001_init_seed_5/checkpoints/00001000000.eqx`

Seed 5 was fixed in advance as the parent seed. It is not the best-scoring seed
— seeds 6 and 7 both reached 0.998 against seed 5's 0.996 — and was deliberately
not chosen on performance.
