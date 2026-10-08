# Final submission audit

## Submission-item status

No assignment brief or rubric was found locally, so this table covers the items
named in the project records and requested for this audit; the official packaging
and naming rules still require comparison with the actual brief.

| Item | Status | Evidence / action |
| --- | --- | --- |
| Source code and vendored upstream dependency | Complete in repository | `scripts/` and `upstream/icl-dynamics/` are present. |
| Saved development analyses and comparisons | Complete for the report chains | Phase 3 matched all 34 selected generation rows to their source analyses. |
| Reserved final-test results | Complete | `results/final_test/final_test_results.json` records 30 unique final checkpoints and `PHASE5_FINAL_TEST.md` summarises the outcome. |
| Final results consolidation | Complete | `FINAL_RESULTS_SUMMARY.md`. |
| `README.md` | Present and current | Final-test and distributed-results wording corrected in this audit. |
| `README.txt` | Missing | Create it if the brief requires that exact filename/format; do not assume `README.md` will be accepted as a substitute. |
| `requirements.txt` | Present | 28 pinned requirements; all installed versions match and `pip check` reports no broken requirements. |
| Extended abstract | Not found | Still needs completion or addition if required. |
| Contribution statement | Not found | Still needs completion or addition if required. |
| NeurIPS ethics questionnaire | Not found | Still needs completion or addition if required. |
| Faculty AI Ethics Statement | Not found | Still needs completion or addition if required. |
| Assignment brief/rubric | Not found | Obtain it and verify the final file list, filenames, length limits and submission format. |

## Repository and reproducibility status

The repository contains the fixed class split, development and final evaluators,
saved configs, logs, checkpoints, generated data, analyses, comparison JSON/PNG
files, the frozen final-test evaluator script and its machine-readable output.
Checkpoint loading portability and analysis-version safeguards are documented in
`PHASE1_FIXES.md`. The report-chain validation is documented in
`PHASE3_VALIDATION.md`.

Reproducibility is strong for inspecting and re-scoring the saved artifacts, but
not fully revalidated from a clean machine in this audit. The active virtual
environment uses Python 3.11.4 while README specifies Python 3.10; its 28 pinned
package versions match `requirements.txt`, and `pip check` passes. A clean
Python-3.10 installation should still be tested before claiming fresh-environment
reproduction. The final test should not be rerun merely for this check.

## Stale and contradictory documentation

Corrected during this audit:

- README statements that the final test had never been generated/scored;
- HANDOVER statements that final testing was outstanding;
- findings 01, 02, 04, 05 and 06, preserving that the test was unrun at those
  stages while recording that it was later scored once after the freeze;
- README statements that `results/` was not distributed, which contradicted the
  tracked repository contents.

`PHASE1_FIXES.md`, `PHASE3_VALIDATION.md` and `PHASE5_SETUP.md` still state that
the final test had not been run *during those phases*. Those are dated provenance
records and should not be rewritten as if the test had already occurred. Their
status is superseded by `PHASE5_FINAL_TEST.md` for the current repository state.

No local evidence supports the phrase “lecturer-approved” for the modified
extended task. The technical rationale is documented, but any approval claim
needs an external approval record or citation.

## Files modified by this audit

- Added `FINAL_RESULTS_SUMMARY.md` and `FINAL_SUBMISSION_AUDIT.md`.
- Updated `README.md` and `context/HANDOVER.md` to reflect the completed frozen
  final-test evaluation and the submitted results tree.
- Updated `findings/01_learning_rate_search.md`,
  `findings/02_recursive_generations.md`, `findings/04_extended_task_tuning.md`,
  `findings/05_label_generation_strategies.md` and
  `findings/06_symbol_distribution_experiments.md` only to distinguish their
  historical pre-final-test state from the current completed state.

## Upstream and file hygiene

- `git diff -- upstream/icl-dynamics` is empty, and the only repository commit
  touching that path is the initial import. It is unmodified relative to the
  current repository HEAD. Because it is vendored without nested Git metadata,
  the README's stated upstream commit was not independently reverified here.
- No tracked `__pycache__`, `.pyc`, backup, scratch or debug artifact was found.
- Two ignored bytecode files exist under `scripts/**/__pycache__/`; they and the
  ignored `.venv/` directory should not be included in an archive assembled
  outside Git. `temporary_checks/` was not present.
- The historical evaluator-size comparison files are explicitly documented
  provenance, not temporary files.
- The working tree was clean before this documentation audit. No experiment
  data, metrics, code, checkpoint, config or dataset was changed.

## Checks performed

- Read the Phase 1, Phase 3, Phase 5 setup and Phase 5 final-test records, README,
  HANDOVER, all findings, requirements and the selected comparison/result JSONs.
- Compared last-generation development values with the frozen final-test JSON.
- Confirmed the final-test JSON contains 30 unique model/run records and final
  checkpoint iteration 1,000,000 for the frozen chains.
- Confirmed `class_splits.json` marks the reserved evaluator as built.
- Searched the repository for an assignment brief and the named submission
  documents; none was found.
- Checked Git status, upstream-path diff/history, ignored/temp candidates and
  pinned installed package versions.
- Ran `.venv\Scripts\python.exe --version` (Python 3.11.4) and
  `.venv\Scripts\python.exe -m pip check` (no broken requirements).
- Did **not** train, generate/regenerate data, rerun experiments, change metrics,
  or run the final test.

## Final checklist

- [ ] Obtain the official assignment brief and reconcile every required item.
- [ ] Create `README.txt` if that literal filename is required.
- [ ] Complete/add the extended abstract and contribution statement.
- [ ] Complete/add the NeurIPS ethics questionnaire and Faculty AI Ethics Statement.
- [ ] Add or cite evidence before calling the extended-task modification
  lecturer-approved.
- [ ] Verify a clean Python 3.10 install from `requirements.txt`.
- [ ] Exclude `.venv/` and `__pycache__/` from any manually created archive.
- [ ] Use `FINAL_RESULTS_SUMMARY.md` and the saved JSONs as the numerical source;
  do not rerun or tune against the reserved final test.
- [ ] Review `git diff` and package only intended tracked files before submission.
