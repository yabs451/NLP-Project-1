# Report figures

## Regeneration

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/make_report_figures.py
```

This writes:

- `results/report_figures/query_accuracy_by_generation.png`

The script reads JSON only. It does not import model code, load checkpoints,
evaluate examples, train models, regenerate data, change metrics, or access the
reserved final-test evaluator.

## Exact data sources

The figure reads the `generations` array from these saved development-comparison
files:

- base `dev_accuracy`:
  `results/base_task/recursive/generation_comparison.json`;
- extended `query_label_accuracy`:
  `results/extended_task/recursive/experiments/label_argmax/generation_comparison.json`;
- `results/extended_task/recursive/experiments/label_sampling_temperature_3/generation_comparison.json`;
- `results/extended_task/recursive/experiments/label_sampling_temperature_5/generation_comparison.json`;
- `results/extended_task/recursive/experiments/context_feedback_temperature_one_third/generation_comparison.json`;
- `results/extended_task/recursive/experiments/symbol_sampling_temperature_0.2/generation_comparison.json`.

## Recommendation

`query_accuracy_by_generation.png` is the recommended figure for the two-page
abstract. It shows the stable base and label-argmax chains, the two label-sampling
collapses, the delayed context-feedback collapse, and the high query accuracy of
next-symbol T=0.2 in one panel. Its y-axis is explicitly development accuracy;
final-test endpoint values should be reported separately from
`results/final_test/final_test_results.json`.

No new scientific computation occurred. The script only reformats already saved
comparison values into a report-oriented figure.
