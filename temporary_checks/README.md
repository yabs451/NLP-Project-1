# temporary_checks — retired evaluation-size comparison

This folder holds one retired piece of work, kept at the user's request. **It is
not part of the main experiment**, nothing under `scripts/` imports it, and no
reproduction step needs it.

## What was retired

The project briefly maintained two development evaluators drawn from the same
100 held-out classes: the 1,000-question set used for everything, and a
10,000-question set used once to check whether the smaller set was precise
enough to choose between models.

It was: both evaluators ranked the five learning rates identically and selected
the same one. The 10,000-question set was therefore retired, and the project now
uses the 1,000-question evaluator throughout.

## What is here

- `rebuild_large_dev_evaluator.py` — recreates the 10,000-question evaluator from
  its recorded seed (3007) and the unchanged development classes. The `.h5`
  itself is not stored, because it is ~60 MB of regenerable data.

## Where the measurements live

The comparison results stay under `results/`, not here, so there is one copy of
each measurement:

- `results/base_task/tuning/evaluator_comparison.json`
- `results/base_task/tuning/evaluator_size_comparison.png`

Both cover **the original 15 candidates only** (learning rates 1e-6 to 1e-4 at
seeds 5, 6, 7). The later 1e-3 candidates were never scored on 10,000 questions.

Written up in `findings/03_learning_rate_search.md`.

## Status

Trackable for now, by decision. Delete this folder when the comparison is no
longer wanted; nothing else will break.

**Note:** if you run the rebuild script, do not commit the resulting
`eval_dev_large.h5` — it is ~60 MB of regenerable data.
