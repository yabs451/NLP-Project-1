# 09 — Comparing label-generation strategies: argmax against temperature-3 sampling

16 September 2026. Local operational report; not tracked in Git.

> **Superseded, and kept as a record of its own stage.** These two chains ran at
> the authors' 1e-05, before the extended task had been tuned. Stage 10 tuned it,
> selected 1e-3, reran the comparison with four conditions, and retired these runs
> to `temporary_checks/extended_task_at_learning_rate_1e-05/`. The measurements
> below are stage 09's own and are left unchanged; the `results/extended_task/...`
> paths it names have moved into that folder, and its conclusions are superseded
> by `findings/05_label_generation_strategies.md`.

## Files I used, and what they gave me

| Source | What it does | What I took from it |
| --- | --- | --- |
| `Development/reports/08_extended_task_experiment.md` | Previous stage | The extended task's design, the sequence layout, the ablation method, and the fact that the argmax chain drifted only slightly |
| `results/extended_task/generation_0/` | The already-trained shared parent | Reused as-is; not retrained, not duplicated |
| `results/extended_task/generation_1..4/` (argmax) | The completed argmax chain | Relocated into the new condition folder; their `analysis.json` records were reused unchanged |
| `scripts/extended_task/extended_model.py` `generate_continuation` | Autoregressive generation | The place the label rule had to become a setting rather than a hard-coded argmax |
| `scripts/extended_task/run_extended.py` | The chain pipeline | Where the folder layout, the seeds and `config.json` are decided |
| `scripts/extended_task/analyse_extended.py` | Per-generation and chain analysis | The measure definitions, head-selection rule and checkpoint selection, all reused unchanged so both conditions are scored identically |
| `upstream/models.py` `call_with_all_aux` | Backbone forward pass | Unchanged; no upstream file was touched at any point this stage |

## Files I changed, and why

| File | Change | Why |
| --- | --- | --- |
| `scripts/extended_task/extended_model.py` | Added `choose_label(logits, strategy, temperature, key)`; `generate_continuation` now takes `label_key`, `label_strategy` and `label_temperature` | One implementation covers both conditions. Argmax remains the default, so the existing behaviour is unchanged unless asked for |
| `scripts/extended_task/run_extended.py` | Shared `generation_0/`, per-condition `experiments/<condition>/`; `--label-strategy` and `--label-temperature`; a separate `label_key` chain from `LABEL_SAMPLING_SEED = 15`; exact error **counts** alongside rates; strategy, temperature, rule and seed written into `config.json` | Two conditions had to be runnable, resumable and distinguishable from their own recorded settings, without a second script |
| `scripts/extended_task/analyse_extended.py` | `--condition`, `--compare-conditions`, a condition-aware `chain_folders`, and a fresh read of `dataset_quality.json` instead of the copy cached inside `analysis.json` | The same analysis has to serve both chains and produce the cross-condition table |
| `findings/03_extended_task_recursion.md` | Rewritten as the single comparison finding | The task and model description was already there; adding a second finding would have repeated the methods |
| `README.md` | Shared generation 0, both run commands, the three analysis commands, plain-language argmax vs temperature 3, updated output paths | The paths moved, so the reproduction instructions had to move with them |
| `Development/reports/08_extended_task_experiment.md` | Two wording corrections (coverage vs frequency balance; the standard-error claim) | Carried over from the documentation-correction pass |
| `results/extended_task/experiments/label_argmax/generation_*/dataset_quality.json` | Backfilled `query_label_errors` / `next_label_errors` by recounting from the saved datasets | The published rates rounded to 0.0000–0.0002 and hid a real monotone accumulation |

No new file was created outside the two write-ups and the temperature-3 run
outputs. No timing files, monitoring files, verification scripts or duplicate
manifests were produced. Both requirements files are untouched. Nothing was
committed, pushed or reset.

## The final structure

```
results/extended_task/
  generation_0/                                   the shared parent, trained once
  experiments/
    label_argmax/
      generation_1 .. generation_4/
      generation_comparison.json, .png
    label_sampling_temperature_3/
      generation_1 .. generation_4/
      generation_comparison.json, .png
  condition_comparison.json, .png                 the two chains side by side
```

Each generation folder holds `config.json`, `log.h5`, `checkpoints/` (55),
`training_data.h5`, `dataset_quality.json` and `analysis.json`.

The argmax generations were **moved**, not copied — there is one copy of each
model on disk. A successor's `config.json` records its parent's full path, so the
two chains cannot be confused on resume, and the pipeline still refuses to write
into an existing finished run.

## The exact generation procedure

For each example, the parent produces three tokens in order, each step
conditioned on what it has already produced:

1. **Query label** — `argmax(logits)`, or `categorical(key, logits / 3)`.
2. **Next symbol** — `categorical(key, symbol_logits)`, i.e. temperature 1, in
   both conditions.
3. **Following label** — by the same rule as step 1.

Temperature divides the **logits** before the softmax. `jax.random.categorical`
applies the softmax internally, so dividing its `logits` argument is exactly
sampling from `softmax(logits / T)`; no already-normalised probability is
rescaled anywhere. All five labels stay eligible: nothing restricts the draw to
the two context labels, no generated mistake is corrected or rejected, and no
true target is mixed in.

Opening contexts come from the authors' original generator at every generation in
both conditions.

Two independent key chains advance in step: the existing `key`
(`GENERATION_SEED = 14`) still drives the symbol sampling, and a new `label_key`
(`LABEL_SAMPLING_SEED = 15`) drives the label sampling. Because the label keys are
a separate chain rather than extra splits of the existing one, adding them left
the opening-question stream and the symbol stream untouched.

**Sampling a query label does change the symbol probabilities.** Generation is
autoregressive, so the sampled label is fed back in as the position-5 token
before the symbol head is read. The symbol *rule* is identical across conditions;
its inputs are not, and the symbol outputs are not expected to match.

Temperature applies only to generating a successor's training data — not to the
training loss, not to evaluation decoding, and not to generation 0's data.

### Verification before running anything new

Before training the temperature-3 chain I re-ran the refactored argmax path
against the saved generation-1 dataset. Opening contexts, query labels, symbol
choices and next labels were **bit-identical** to what was already on disk, so
the refactor did not change the completed condition. The check lived in
`Development/` and was deleted afterwards.

That check initially reported a mismatch on the opening contexts. The code was
right and the check was wrong: `draw_opening_contexts(2000)` returns 1,984 rows
(62 batches of 32), while the real run's first generation batch is 2,000, and
JAX's `categorical` gives different draws for different array shapes. Drawing a
slightly larger block and slicing to 2,000 made it pass.

## Results

Both chains completed. All losses stayed finite; 55 checkpoints per generation;
12 of them analysed per generation, at the same points in both conditions
(0, 1,024, 2,016, 5,024, 10,016, 20,000, 40,000, 80,000, 160,000, 280,000,
540,000, 1,000,000).

**What the parents wrote down** (wrong labels per 1,000,000 examples, against the
original context mapping):

| Dataset for gen | argmax query / next | T3 query / next |
| --- | ---: | ---: |
| 1 | 19 / 41 | 147,465 / 155,856 |
| 2 | 43 / 134 | 570,974 / 582,900 |
| 3 | 78 / 175 | 763,252 / 767,131 |
| 4 | 117 / 224 | 789,254 / 790,321 |

**What the successors learned** (same fixed 1,000-question development
evaluator, teacher forced, final checkpoints):

| Gen | argmax query acc / loss | T3 query acc / loss | argmax induction | T3 induction |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.962 / 0.104 | 0.962 / 0.104 | 0.7722 | 0.7722 |
| 1 | 0.963 / 0.100 | 0.951 / 0.279 | 0.7662 | 0.6922 |
| 2 | 0.961 / 0.099 | 0.601 / 1.182 | 0.7633 | 0.0959 |
| 3 | 0.959 / 0.103 | 0.501 / 1.462 | 0.7596 | 0.0036 |
| 4 | 0.957 / 0.105 | 0.366 / 1.561 | 0.7554 | 0.0041 |

Symbol loss stayed near ln 2 ≈ 0.6931 in both chains (argmax 0.6967–0.6993,
slightly above it; temperature 3 0.6924–0.7001), and all 50 training classes
appeared as next symbols in every generated dataset.

The argmax condition's numbers were **not recomputed**. Its comparison table was
rebuilt from the existing `analysis.json` records, and every value matches what
was previously published. The only argmax files that changed content are the four
`dataset_quality.json` files, which gained exact error counts.

## Interpretation

**The label-generation strategy decided the outcome in these two chains.** Argmax
labels held the chain steady (0.962 → 0.957 over four generations);
temperature-3 labels brought it to 0.366, above the 0.2 that uniform guessing over
five labels would give but far below the parent. This is degradation rather than
stability, in one chain per condition at one seed and budget. *(It was the
clearest degradation in the project when this was written; stage 10 produced
larger effects at the tuned rate.)*

**A route is visible in the data, as a proposed explanation rather than a tested
one.** At temperature 3 the parent writes a label it did not think most likely a
large fraction of the time, and that becomes the successor's target. The figures
15% → 57% → 76% → 79% are the **error rates of the generated datasets**, not the
successors' own scores; those are in the table above. That the one caused the
other is the obvious reading, but this stage did not test it.

**On whether circuit change precedes accuracy decline, generation 1 is the only
informative point and it is suggestive, not conclusive.** From generation 0 to
temperature-3 generation 1, query accuracy fell 1.1% relative while the strongest
induction score fell 10.4% relative. But the two are on different scales, so a
percentage comparison between them is not a like-for-like test; query *loss* also
rose 169% at the same step, so behaviour was not intact while only the mechanism
moved; and this is one generation in one chain.

**What the numbers do not establish.** A drop in an attention score is not a
demonstrated loss of circuit function — ablating one head still costs 5.6–8.4
percentage points at temperature-3 generations 1–4, a similar absolute cost from
a much lower base, and a single-head ablation is not a measure of the whole
circuit's importance. All 50 classes appearing is retained coverage, not
unchanged relative frequencies. And the comparison as a whole is argmax versus
temperature-3 sampling: it does not isolate temperature.

## Remaining limitations and problems

1. **Not a temperature experiment.** Sampling at temperature 1 was not run, so
   nothing here separates "sampling rather than argmax" from "temperature 3
   rather than a lower temperature".
2. **The selected head identity changed in the temperature-3 chain** (induction
   L1H2 → L1H3 → L1H4; previous-token L0H5 → L0H2 → L0H5). Heads are selected
   from each model's own scores, so its ablation column compares different heads
   across generations and is not a like-for-like series.
3. **One chain per condition.** One initialisation seed, one generation seed, one
   budget, one development set, one learning rate (1e-5, untuned for this task).
4. **Generation 1 is a single data point** for the ordering question, and the
   measures involved are on different scales.
5. **12 of 55 checkpoints analysed**, so the within-training trajectories are at
   those points only. The saved checkpoints for the finer analysis exist if it is
   wanted later.
6. **The reserved final-test classes have still never been generated or scored**,
   as intended at this stage.

## Where things are

Paths as they were at the end of stage 09. Everything listed here now sits inside
`temporary_checks/extended_task_at_learning_rate_1e-05/`, whose README gives the
current layout; the README's extended-task section describes the stage-10
experiment instead.

- Finding: `findings/03_extended_task_recursion.md`
- Cross-condition numbers and figure: `results/extended_task/condition_comparison.json`, `.png`
- Per-chain tables and figures: `results/extended_task/experiments/<condition>/generation_comparison.json`, `.png`
- Shared parent: `results/extended_task/generation_0/`
- Commands: README, "Extended task" section
