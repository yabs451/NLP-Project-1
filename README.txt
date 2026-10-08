NLP Project 1: submission reproduction guide
============================================

Environment
-----------
Run commands from the repository root on Windows PowerShell. The maintained
environment is Python 3.10 on CPU. Create the environment and install the pinned
dependencies with:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --timeout 120 -r requirements.txt
    .\.venv\Scripts\python.exe -m pip check

Repository structure
--------------------
    scripts/                  project training, evaluation and analysis code
    scripts/base_task/        base-task tuning, recursion and analysis
    scripts/extended_task/    extended-task tuning, recursion and analysis
    findings/                 scientific findings
    results/                  saved data, checkpoints, analyses and figures
    context/HANDOVER.md       detailed experiment inventory and conventions
    upstream/icl-dynamics/    vendored, unmodified upstream implementation

Evaluation data
---------------
The deterministic split and fixed development evaluator are prepared with:

    .\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py

This uses the class IDs and seeds recorded in
results/evaluation_data/class_splits.json. The disjoint reserved final evaluator
was built once with --build-final-test after the experiment design was frozen.
It already exists and should not be regenerated for ordinary reproduction.

Base task
---------
The complete learning-rate search is:

    .\.venv\Scripts\python.exe scripts/base_task/tune_learning_rate.py

Train generation 0 at the selected learning rate, then its four successors:

    .\.venv\Scripts\python.exe scripts/base_task/train_original.py --results-subfolder recursive --run-name generation_0 --learning-rate 0.001 --init-seed 5
    .\.venv\Scripts\python.exe scripts/base_task/run_recursive.py --parent results/base_task/recursive/generation_0 --generations 4

Regenerate each run analysis, then the cross-generation comparison. Repeat the
first command for generation_0 through generation_4.

    .\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py results/base_task/recursive/generation_0
    .\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py --compare results/base_task/recursive

Extended task
-------------
Run the separate extended-task learning-rate search:

    .\.venv\Scripts\python.exe scripts/extended_task/tune_extended.py

The nine maintained recursive conditions are reproduced with:

    .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --generations 4
    .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --label-strategy sample --label-temperature 1 --generations 4
    .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --label-strategy sample --label-temperature 3 --generations 4
    .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --label-strategy sample --label-temperature 5 --generations 4
    .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --context-temperature 1 --generations 4
    .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --context-temperature 1/3 --generations 6
    .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --context-temperature 0.2 --generations 6
    .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --next-symbol-temperature 1/3 --generations 6
    .\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --next-symbol-temperature 0.2 --generations 6

Generation 0 is trained once and shared. In run_extended.py, --generations names
the final generation. In the base recursive script it means the number of new
successors. Do not combine label sampling, context feedback and non-default
next-symbol temperature in one command.

Extended analysis and existing figures
--------------------------------------
Analyse each condition, then regenerate the three family comparisons:

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

These analysis commands use saved checkpoints and development data; they do not
train models. They regenerate the analysis JSON and figures described in
README.md. The canonical extended previous-token metric uses positions 1 and 3.
Positions 1, 3 and 5 are historical diagnostic values only.

The report-only query-accuracy figure reads saved comparison JSON and is rebuilt
without loading a model:

    .\.venv\Scripts\python.exe scripts/make_report_figures.py

Frozen final test
-----------------
scripts/evaluate_final_test.py checks that eval_final_test.h5 exists, loads only
the final saved checkpoint for each frozen report-chain model, applies the same
predictive scoring definitions used in development, and writes one file:

    results/final_test/final_test_results.json

The frozen evaluation has already been completed. Its results are recorded in
that JSON and PHASE5_FINAL_TEST.md. Do not rerun it for tuning, model selection,
or exploratory analysis. The historical command is:

    .\.venv\Scripts\python.exe scripts/evaluate_final_test.py

Computational cost
------------------
The tuning scripts, train_original.py, run_recursive.py and run_extended.py train
models and are computationally expensive. prepare_evaluation_data.py writes
evaluation datasets. Per-run analysis loads many saved checkpoints and can also
be slow, although it performs no training. Family comparisons and
make_report_figures.py read existing analysis files and are inexpensive. For
submission review, use the saved results instead of rerunning expensive steps.

See README.md, context/HANDOVER.md, FINAL_RESULTS_SUMMARY.md and
FINAL_SUBMISSION_AUDIT.md for the full methodology, provenance and limitations.
