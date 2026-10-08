# Phase 3 validation

## Scope

Checked existing saved results only:

- base recursive generations 0-4;
- extended `label_argmax` generations 0-4;
- extended `label_sampling_temperature_3` generations 0-4;
- extended `label_sampling_temperature_5` generations 0-4;
- extended `context_feedback_temperature_one_third` generations 0-6;
- extended `symbol_sampling_temperature_0.2` generations 0-6.

All 34 generation rows matched their source per-generation analysis records. For
extended runs, the embedded training-data quality also matched each saved
`dataset_quality.json`. Every selected extended analysis is version 2 and names
positions 1 and 3 as the canonical previous-token metric.

## Key validated results

Values are generation 0 → last generation. Losses are final query / following /
symbol loss in nats. Previous-token is the canonical positions-1-and-3 score.

| chain | query accuracy | following accuracy | final losses | wrong generated query labels | induction | previous-token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base, g0-4 | 0.996 → 0.996 | — | 0.011 / — / — | 0 in every successor | 0.960 → 0.960 | 1.000 → 1.000 |
| label argmax, g0-4 | 0.997 → 0.993 | 0.996 → 0.997 | 0.046 / 0.015 / 0.693 | 13 at g4 | 0.977 → 0.979 | 1.000 → 1.000 |
| label sampling T=3, g0-4 | 0.997 → 0.489 | 0.996 → 0.456 | 1.556 / 1.558 / 0.692 | 788,744 at g4 | 0.977 → 0.021 | 1.000 → 0.694 |
| label sampling T=5, g0-4 | 0.997 → 0.194 | 0.996 → 0.186 | 1.608 / 1.607 / 0.695 | 799,069 at g4 | 0.977 → approximately 0 | 1.000 → 0.731 |
| context feedback T=1/3, g0-6 | 0.997 → 0.410 | 0.996 → 0.399 | 2.474 / 2.692 / 0.694 | 0 at every generation | 0.977 → 0.129 | 1.000 → 1.000 |
| next-symbol T=0.2, g0-6 | 0.997 → 0.997 (minimum 0.989) | 0.996 → 0.940 (minimum 0.898) | 0.033 / 0.281 / 5.746 | g1-6: 0, 61, 21, 418, 308, 231 | 0.977 → 0.646 | 1.000 → 0.692 |

## Canonical versus historical previous-token analysis

The retained positions-1/3/5 arrays reproduce the historical Git values. Head
selection changed in only three selected model generations:

| model | historical head → canonical head | historical → canonical ablation effect |
| --- | --- | ---: |
| label sampling T=3, g2 | H4 → H0 | +5.4 → +0.4 percentage points |
| next-symbol T=0.2, g2 | H4 → H2 | +8.9 → +1.0 points |
| next-symbol T=0.2, g3 | H5 → H7 | +0.6 → +1.5 points |

All other selected head choices and ablation effects were unchanged. The
next-symbol-T=0.2 final generation remains H3 with a +12.7-point effect. Thus a
few individual ablation numbers change materially, but the defensible
interpretation does not: effects vary by model and selected head, and a
single-head final-checkpoint ablation does not establish circuit necessity or
redundancy.

Attention and predictive performance were measured across the selected
checkpoints. Head ablations were performed only at each model's final checkpoint.
Positions 1 and 3 are canonical; positions 1/3/5 are retained only as the named
historical extended diagnostic.

## Conclusions

The main conclusions do not change:

- the base chain is stable across generations 0-4;
- label argmax remains broadly stable;
- temperature-3 and temperature-5 label sampling strongly degrade both label
  outputs and corrupt large fractions of generated query targets;
- context feedback at temperature 1/3 eventually degrades strongly despite zero
  recorded generated query-label errors throughout the chain;
- next-symbol temperature 0.2 strongly degrades symbol loss and the induction
  attention score while query accuracy remains high; following-label accuracy is
  not fully stable and dips to 0.898;
- predictive accuracy, attention-based induction measures, and symbol behaviour
  do not necessarily degrade together.

These results do **not** support claiming that attention changes precede accuracy
changes, that an induction circuit disappeared, that context concentration caused
the context-feedback collapse, that both label outputs stayed intact under
next-symbol temperature 0.2, or that the results generalise across seeds, chains,
the full question distribution, or the reserved final test.

## Files modified in Phase 3

- `findings/05_label_generation_strategies.md` — replaced selected historical
  previous-token values/head statements with canonical 1/3 results and recorded
  the one changed selected ablation.
- `findings/06_symbol_distribution_experiments.md` — updated selected canonical
  previous-token values and the two changed next-symbol-T=0.2 head selections.
- `PHASE3_VALIDATION.md` — this validation record.

No saved analysis, comparison, checkpoint, config, or dataset file was modified
by Phase 3.

## Checks run

- Read the requested project documents and selected comparison, per-generation
  analysis, and dataset-quality JSON files.
- Cross-checked all selected comparison fields against their source analysis:
  query/following accuracy, query/following/symbol loss, query-error counts,
  induction maximum, and canonical previous-token maximum. Result: 34/34 rows
  matched.
- Compared every selected canonical previous-token array/head/ablation with its
  committed historical counterpart using `git show HEAD:<analysis.json>`.
- Ran `git diff --check` after the documentation edits.

Models trained: **NO**. Datasets generated or regenerated: **NO**. New experiments:
**NO**. Reserved final-test evaluation: **NO**.
