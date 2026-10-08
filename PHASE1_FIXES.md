# Phase 1 cleanup

## Summary

Phase 1 makes the canonical extended previous-token metric use token positions 1
and 3, keeps the former 1/3/5 value as an explicitly named diagnostic, versions
extended analysis outputs, and refuses to reuse or compare caches made with the
old definition. It also corrects final-checkpoint-only ablation wording, repairs
legacy base-config input paths in memory, restricts base comparison discovery to
numbered generation directories, and rejects commands that mix extended
experiment families.

No scientific experiment design, checkpoint, saved historical config, training
dataset, or upstream source file was changed.

## Modified files

- `scripts/extended_task/analyse_extended.py`
  - Changed the canonical previous-token average from positions 1/3/5 to positions
    1/3. This canonical value now selects the previous-token head used for the
    main ablation.
  - Preserved positions 1/3/5 under the explicit
    `previous_token_scores_positions_1_3_5_diagnostic_layer0` name.
  - Added `analysis_version: 2` and exact metric-definition strings to new
    per-generation and comparison JSON.
  - Reuses a per-generation cache only when its version and metric definition
    match. Family comparison now errors clearly when a condition table is old.
    This prevents an unversioned 1/3/5 cache from being treated as 1/3.
- `scripts/extended_task/run_extended.py`
  - Added a small validation rule that rejects any command combining label
    sampling, non-reference next-symbol temperature, and/or context feedback.
    One family may vary at a time; no new condition-naming scheme was introduced.
- `scripts/common.py`
  - `load_run_options` now repairs missing saved `data_file` and
    `load_eval_data` paths in memory using this checkout's feature and evaluation
    files. Historical `config.json` files are not rewritten.
- `scripts/base_task/analyse_runs.py`
  - Base `--compare` now accepts only directories named `generation_<integer>`.
    Files such as `generation_comparison.json` and `.png` are ignored.
- `README.md`
  - States that attention/performance are measured over selected checkpoints but
    ablations are final-checkpoint-only.
  - Documents the 1/3 canonical metric, the historical 1/3/5 definition, cache
    versioning, and mixed-family rejection.
  - Corrects next-symbol-temperature and context-feedback provenance wording.
- `context/HANDOVER.md`
  - Corrects the ablation checkpoint claim and adds the metric-definition history
    and provenance limitation.
- `findings/05_label_generation_strategies.md`
  - Corrects ablation coverage to final checkpoints only and marks quoted
    previous-token results and selected previous-token-head ablations as
    historical 1/3/5 results.
- `findings/06_symbol_distribution_experiments.md`
  - Makes the same ablation and historical-metric corrections.
- `results/extended_task/recursive/generation_0/analysis.json`
  - Recomputed from its existing checkpoints with analysis version 2. Contains
    both canonical 1/3 and named diagnostic 1/3/5 values.
- `results/extended_task/recursive/experiments/label_argmax/generation_1/analysis.json`
- `results/extended_task/recursive/experiments/label_argmax/generation_2/analysis.json`
- `results/extended_task/recursive/experiments/label_argmax/generation_3/analysis.json`
- `results/extended_task/recursive/experiments/label_argmax/generation_4/analysis.json`
  - Each was recomputed from existing checkpoints with the version-2 metric
    fields and final-checkpoint head selection based on positions 1/3.
- `results/extended_task/recursive/experiments/label_argmax/generation_comparison.json`
- `results/extended_task/recursive/experiments/label_argmax/generation_comparison.png`
  - Regenerated from the version-2 `label_argmax` analyses.
- `PHASE1_FIXES.md`
  - This audit trail, check record, limitation statement, and reproduction guide.

The base comparison check rewrote its existing JSON/PNG deterministically, but
their content hashes remained identical to `HEAD`; they are therefore not listed
as content modifications.

## Historical analysis after the metric correction

Any extended analysis produced before `analysis_version: 2` used positions 1, 3
and 5 for the field then called the previous-token score. Those values must not be
relabeled as 1/3 results. The following are historical until regenerated:

- all unversioned per-generation `analysis.json` files outside the refreshed
  shared generation 0 and `label_argmax` generations 1-4;
- the corresponding unversioned condition `generation_comparison.json` files and
  figures;
- the three family comparisons and figures built from those condition tables;
- previous-token values quoted in findings 05 and 06;
- previous-token-head ablation results whose head was selected by the old 1/3/5
  score.

The label-argmax analyses were refreshed in Phase 1. Their version-2 JSON retains
the old 1/3/5 value as a diagnostic, so no old value was silently relabeled.
Induction metrics and induction-head selection were not changed by this cleanup.

## Commands/checks run and outcomes

1. Required source review: read `context/HANDOVER.md`, `README.md`, all files under
   `scripts/base_task/` and `scripts/extended_task/`, and the relevant shared
   loader in `scripts/common.py`. `p1_audit.md` was checked first but does not
   exist in this checkout.
2. Python syntax parse:

   ```powershell
   .\.venv\Scripts\python.exe -c "import ast, pathlib; files=['scripts/common.py','scripts/base_task/analyse_runs.py','scripts/extended_task/run_extended.py','scripts/extended_task/analyse_extended.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8'), filename=f) for f in files]; print('syntax ok:', ', '.join(files))"
   ```

   Outcome: all four modified Python files parsed successfully.
3. Legacy base checkpoint loading:

   ```powershell
   .\.venv\Scripts\python.exe -c "from pathlib import Path; import sys; sys.path.insert(0,'scripts'); import common; folder=Path('results/base_task/recursive/generation_0'); opts=common.load_run_options(folder); assert Path(opts.data_file).resolve()==common.FEATURE_FILE.resolve(); assert Path(opts.load_eval_data[0]).resolve()==common.DEV_EVALUATOR_FILE.resolve(); iteration=common.available_checkpoints(folder)[-1]; common.load_checkpoint(folder,iteration,opts); print('loaded',iteration)"
   ```

   Outcome: loaded checkpoint 1,000,000. The config's missing absolute paths from
   `C:\Users\13327\...` resolved to this repository's feature and evaluator files.
4. Base comparison with existing output files present:

   ```powershell
   .\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py --compare results/base_task/recursive
   ```

   Outcome: succeeded for generations 0-4 while the folder already contained
   `generation_comparison.json` and `.png`. The rewritten files were byte-identical
   to the tracked versions.
5. Invalid mixed extended settings:

   ```powershell
   .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --label-strategy sample --next-symbol-temperature 0.2 --generations 0
   ```

   Outcome: exited with status 2 before loading data or training, with the clear
   error `incompatible experiment families in one command`.
6. Previous-token position calculation: ran a synthetic-attention check through
   `attention_measures` with corrected scores 0.8 at position 1, 0.6 at position
   3, and -0.08 at position 5.

   Outcome: canonical score was 0.70 = mean(positions 1,3); the separately named
   historical diagnostic was 0.44 = mean(positions 1,3,5).
7. Existing-analysis recomputation, without training:

   ```powershell
   .\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition label_argmax
   ```

   Outcome: successfully recomputed shared generation 0 and label-argmax
   generations 1-4 from their saved checkpoints. Every per-generation record now
   has `analysis_version: 2`, the 1/3 metric definition, and the named 1/3/5
   diagnostic. No checkpoint or training dataset changed.
8. Stale-cache family-comparison guard:

   ```powershell
   .\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --compare-family label
   ```

   Outcome: intentionally refused to combine the refreshed label-argmax table
   with the unversioned `label_sampling_temperature_1` table and instructed the
   caller to rerun that condition first.
9. Patch whitespace check:

   ```powershell
   git diff --check
   ```

   Outcome: passed with no whitespace errors. Git printed only the repository's
   normal LF-to-CRLF working-tree warnings.

## Unverified or unresolved

- `p1_audit.md` was not present locally, so it could not be read.
- Only the `label_argmax` chain was recomputed as the end-to-end proof. The other
  eight extended conditions and the three family comparisons remain historical
  and intentionally cannot be mixed with version-2 output until regenerated.
- Previous-token results and previous-token-head ablations quoted in findings 05
  and 06 were not numerically rewritten; they are explicitly labeled historical.
- Checks used the existing `.venv`, which reports Python 3.11.4 although project
  documentation specifies Python 3.10. The environment was not changed.
- The reserved final test was not run.

## Training and data regeneration

- Models trained: **NO**.
- Training datasets regenerated: **NO**.
- Evaluation datasets regenerated: **NO**.
- New experiments run: **NO**.
- Existing checkpoints used for inference-only analysis: **YES**.

## Commands to reproduce Phase 1 and refresh affected analysis

Run the syntax, checkpoint-loading, base-comparison, invalid-mix, and
label-argmax commands above from the repository root. To verify a refreshed
record directly:

```powershell
.\.venv\Scripts\python.exe -c "import json, pathlib; p=pathlib.Path('results/extended_task/recursive/generation_0/analysis.json'); d=json.loads(p.read_text()); assert d['analysis_version']==2; assert d['previous_token_metric_definition'].endswith('positions 1 and 3'); assert 'previous_token_scores_positions_1_3_5_diagnostic_layer0' in d; print('versioned 1/3 analysis with separate 1/3/5 diagnostic')"
```

To regenerate every affected extended analysis from existing checkpoints only
(no training and no dataset generation):

```powershell
$conditions = @(
  'label_argmax',
  'label_sampling_temperature_1',
  'label_sampling_temperature_3',
  'label_sampling_temperature_5',
  'context_feedback_temperature_1',
  'context_feedback_temperature_one_third',
  'context_feedback_temperature_0.2',
  'symbol_sampling_temperature_one_third',
  'symbol_sampling_temperature_0.2'
)
foreach ($condition in $conditions) {
  .\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition $condition
  if ($LASTEXITCODE -ne 0) { throw "analysis failed: $condition" }
}
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --compare-family label
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --compare-family context
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --compare-family symbol
```

Those commands overwrite derived analysis JSON/figures with version-2 results;
they do not train models, alter checkpoints, or regenerate training/evaluation
datasets. Do not run any final-test command as part of Phase 1.
