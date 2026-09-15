# 02 — Code structure and the assignment validation protocol

Date: 15 September 2026. Scope: recovery of the working environment after a
fresh clone, a full map of the code we maintain and the code we borrow, and a
separate assignment evaluation protocol that reserves an untouched final test.
No recursive training, extended task or data-mixture experiment is implemented
here.

## Outcome

The workspace was recovered from the repository alone. The virtual environment
was rebuilt from `requirements-lock.txt` and reproduced **every metric reported
in report 01 exactly** (for example `fsl_train` 21.5% → 28.3%, context-restricted
51.2%), which independently confirms both the previous agent's trial and our
restored environment.

A `--protocol assignment` option now selects a development evaluator built from
100 previously unused character classes, and stops the authors' repeatedly
inspected test-class evaluator from running. A second, disjoint 100-class final
test is specified and recorded but deliberately has no data generated and no
metric computed. In the case we checked, switching protocols left
training unchanged: a 3,200-sequence reproduction run and a 3,200-sequence
assignment run produced checkpoints with identical SHA-256 hashes.

No upstream source file was modified. The protocol is implemented entirely
through options the authors already provide.

## Recovery after the fresh clone

What the clone actually contained, checked rather than assumed:

| Item | Expected | Found |
| --- | --- | --- |
| Project files | README, `.gitignore`, requirements ×2, 3 scripts, report 01 | All present |
| Upstream source | `upstream/icl-dynamics/` | Present, 35 files, 28 MB |
| Feature file | `omniglot_resnet18_randomized_order_s0.h5` | Present, 16,638,016 bytes |
| `.venv/` | ignored | **Absent — rebuilt** |
| `results/` | ignored | **Absent — trial checkpoints, logs and evaluator data gone** |

**Upstream is vendored, not a submodule.** There is no `.gitmodules` and no
nested `.git` directory; all 35 upstream files, including the 16.6 MB feature
file, are tracked directly in our repository. Nothing needed restoring from
GitHub, and there is no incomplete nested-repository reference to repair. The
trade-off is that `git log` in `upstream/icl-dynamics/` is unavailable, so the
upstream commit `85b895c720844e795734b9391b32d2619065f9a1` is now a recorded
claim from report 01 rather than something we can re-verify locally. The
inspection script's `upstream_commit` field consequently cannot run against a
vendored copy; this is noted as a known limitation rather than patched.

Environment rebuild reproduced the recorded versions exactly (`pip check`
reports no broken requirements): JAX/jaxlib 0.4.26, Equinox 0.11.4, Optax 0.2.2,
NumPy 1.26.4, h5py 3.11.0, on CPython 3.10.11, CPU (`TFRT_CPU_0`). We added
**matplotlib 3.8.4** and **tqdm 4.66.4**, which are required to import the
authors' own analysis module `visualize_runs.py`; they are not needed for
training.

The missing short-trial checkpoints were **not** recreated as a prerequisite —
full training starts from the original initialization. Two short runs were made
anyway as protocol checks (below), and these incidentally reproduced report 01's
numbers. Disk space was checked before the full run: 69 GB free against an
estimated 810 MB for the original checkpoint schedule, so the authors' schedule
was kept unchanged.

## Project structure

```
NLP-Project-1/
├── README.md                     project overview, setup and commands
├── .gitignore                    excludes env/caches/results, keeps the split record
├── requirements.txt              compatibility pins we chose
├── requirements-lock.txt         full installed dependency lock
├── reports/
│   ├── 01_setup_and_baseline_trial.md
│   ├── 02_code_structure_and_validation.md   (this file)
│   └── 03_full_baseline_training_and_analysis.md
├── scripts/                      all code we wrote
│   ├── run_baseline.py           launcher; --protocol reproduction|assignment
│   ├── inspect_baseline.py       data/environment/sampler inspection
│   ├── verify_trial.py           checkpoint reload and evaluation replay
│   ├── make_assignment_evaluators.py   class selection + dev evaluator
│   └── analyse_baseline.py       curves, progress measures, attention, ablation
├── upstream/icl-dynamics/        authors' code, unmodified (vendored)
└── results/                      generated; only class_splits.json is tracked
    ├── assignment/               class_splits.json, eval_dev.h5
    └── <run>/                    config, log.h5, checkpoints/, console.log, analysis/
```

### Files we maintain

| File | Purpose | Key functions | Inputs | Outputs | When used |
| --- | --- | --- | --- | --- | --- |
| `scripts/run_baseline.py` | Reads the authors' baseline command out of their shell script and launches their `main.py` as a subprocess | `baseline_arguments()` extracts and de-templates the command; `assignment_arguments()` swaps the evaluator set; `replace_list_option()` edits one option's values; `main()` runs and times the child process | `ih_paper_runs.sh`, CLI flags `--full`, `--protocol`, `--run-name`, `--dry-run` | `results/<run>/` with `command.json`, `console.log`, `timing.json`; upstream writes `config.json`, `log.h5`, `checkpoints/` | Every training run |
| `scripts/inspect_baseline.py` | Records what the data and configuration actually are, and checks the sampler produces the intended task | `main()` | Feature `.h5`, baseline arguments | `results/inspection/inspection.json`, `baseline_options.json` | Once per environment, or after changing data assumptions |
| `scripts/verify_trial.py` | Reloads checkpoints and proves training and saving worked | `load_checkpoint()`, `main()` | A run folder (`config.json`, `log.h5`, `eval_data.h5`, `checkpoints/`) | `verification.json` | After a run that saved evaluator data |
| `scripts/make_assignment_evaluators.py` | Fixes the assignment class splits and builds the development evaluator | `load_baseline_opts()`, `select_classes()`, `build_evaluator()`, `check_evaluator()` | Feature `.h5`, baseline arguments | `results/assignment/class_splits.json` (tracked), `eval_dev.h5` | Once, before any assignment run; re-runnable deterministically |
| `scripts/analyse_baseline.py` | Learning curves, induction-circuit progress measures, attention maps, head ablations | `previous_token_scores()`, `induction_scores()`, `measure_checkpoint()`, `ablation()`, `plot_*()` | A run folder plus `results/assignment/eval_dev.h5` | `results/<run>/analysis/` with `analysis.json`, `curves.png`, `head_measures.png`, `attention_example.png` | After a training run |

Generated artefacts are grouped rather than listed individually. Each run folder
holds one `config.json`, `command.json`, `console.log`, `timing.json` and
`log.h5`, plus a `checkpoints/` directory containing one `.eqx` file per
checkpoint (about 809 KB each; 1,001 files for the original schedule), and an
`analysis/` directory after analysis. `results/inspection/` and
`results/assignment/` each hold one small JSON record plus supporting files.

### Upstream files and functions we rely on

All paths are under `upstream/icl-dynamics/`.

| Concern | Where | What it gives us |
| --- | --- | --- |
| Sampling | `samplers.py` — `get_constant_burst_seq_idxs` (:32), `get_exemplar_inds` (:167), `fewshot_relabel` (:221), `make_data_sampler` (:288) | How a sequence is built: query class, distractor, burstiness, which exemplar, and the per-sequence label reassignment. `make_data_sampler` composes the three into one callable, which is what both training and every evaluator use. |
| Architecture | `models.py` — `SequenceClassifier` (:596), `Transformer` (:472), `TransformerBlock` (:365), `AttentionBlock` (:260), `apply_rope` (:234) | The attention-only two-layer model. `SequenceClassifier` interleaves symbols and labels into 5 tokens (:748-753): even positions are symbols, odd are labels, and the query label is dropped, so the target can never leak into the input. `AttentionBlock` caches `attn_scores`, which is what all our mechanistic measures read. |
| Configuration | `main_utils.py` — `create_parser` (:20), `check_opts` (:119), `get_splits_from_opts` (:153), `get_model_from_opts` (:222), `get_optimizer_from_opts` (:250) | Every default we inherit. `get_splits_from_opts` defines the class/exemplar/label-pair splits as contiguous index ranges, which is the function our protocol builds on. |
| Training | `main.py` — `run_with_opts` (:215), `train_step` (:87), `compute_loss` (:70), `ce` (:59) | The training loop, loss (query-position cross-entropy only) and checkpoint schedule. |
| Evaluation | `main.py` — evaluator construction (:371-400), `evaluate` (:200), `eval_step` (:132), `make_batched_fn` (:183), evaluator loading (:357-369) | How fixed evaluation sets are drawn once and reused, and how `--load_eval_data` injects externally built evaluators — the mechanism our protocol uses. |
| Interpretation | `visualize_runs.py` — `make_forward_fn` (:701), `update_prev_token_over_time` (:338), `plot_prev_token_over_time` (:355), `plot_attention_over_time` (:459); `opto.py` — `make_fn_from_opts` (:150), ablation application (:368-372) | The forward pass that returns attention scores alongside metrics, the authors' definitions of the two progress measures, and the head-ablation mechanism. |

### How our launcher reaches the authors' implementation

`run_baseline.py` does not import the training code. It reads
`ih_paper_runs.sh`, takes the **first** `python main.py` line (the Figure 3a
`omniglot50_rl5` baseline), substitutes the shell variables `$MAIN_RUN_ITERS`,
`$INIT_SEED` and `$SAVE_FOLDER`, repoints `--data_file` at the vendored feature
file, appends `--base_folder results/ --run <name>`, and then runs
`sys.executable -u upstream/icl-dynamics/main.py <args>` as a subprocess with
`JAX_PLATFORMS=cpu` and `WANDB_MODE=disabled`. Inside that process,
`main.run_with_opts` builds splits, samplers, evaluators, the model and the
optimizer from those arguments. The exact argument vector is printed and saved
to `command.json`, so a run is reproducible from its own folder.

The other two analysis scripts take the opposite route: they put
`upstream/icl-dynamics/` on `sys.path` and import `main`, `main_utils`,
`samplers`, `opto` and `visualize_runs` directly, so that evaluation and
mechanistic measurement use the authors' own functions rather than
reimplementations.

## Version control status

| Category | Contents |
| --- | --- |
| Tracked and uploaded | `README.md`, `.gitignore`, `requirements.txt`, `requirements-lock.txt`, `reports/01_...md`, `scripts/{run_baseline,inspect_baseline,verify_trial}.py`, and all 35 upstream files including the 16.6 MB feature file |
| Tracked, modified, not yet committed | `.gitignore`, `scripts/run_baseline.py` |
| Untracked, intended to be committed | `scripts/make_assignment_evaluators.py`, `scripts/analyse_baseline.py`, `reports/02_...md`, `reports/03_...md`, `results/assignment/class_splits.json` |
| Ignored | `.venv/`, `.cache/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `wandb/`, and everything under `results/` except `results/assignment/class_splits.json` |

**Local tracking versus verified upload.** The single commit `21a7df0`
("Initial commit for NLP Project 1") is the only commit. `git ls-remote origin`
returns `21a7df02e91f...` for `refs/heads/main`, so that commit is confirmed
present on GitHub at `https://github.com/yabs451/NLP-Project-1.git`, not merely
committed locally. Everything produced in this stage is newer than that commit
and therefore exists **only on this machine**. As instructed, nothing was
committed, pushed, reset, or repointed.

**Reproducibility issue found and fixed.** The original rule ignored `results/`
wholesale, which also hid `results/assignment/class_splits.json` — the file that
records which class IDs are validation and which are reserved for the final
test. A protocol that is not version-controlled cannot be audited, and
regenerating it depends on the script and seed staying unchanged. `.gitignore`
now excludes `results/*` but re-includes that one record:

```
results/*
!results/assignment/
results/assignment/*
!results/assignment/class_splits.json
```

Verified with `git check-ignore`: the split record is trackable, while
`eval_dev.h5`, run logs and checkpoints remain ignored.

**Reproducibility issue accepted, not fixed.** Ignoring `results/` means that
checkpoints and raw metrics do not survive a fresh clone — which is exactly what
happened to the first trial. This is the right trade-off at ~810 MB per run, but
it means *experimental evidence lives only on the machine that produced it*.
Anything a report depends on must therefore be quoted in the report itself, or
be cheap to regenerate. This stage's reports quote their numbers for that
reason.

## The assignment evaluation protocol

### Why a new protocol is needed

The authors' four evaluators (report 01) are training diagnostics, not a
model-selection protocol. `fsl_test_class` scores the last 100 class rows
throughout training, so those rows have already been inspected repeatedly and
cannot serve as an untouched final test. Meanwhile the middle 1,473-row class
partition is used by **no** original evaluator, which leaves ample untouched
material.

### Decisions

1. **Training is untouched.** The original 50 training classes, exemplar 0, two
   symbol–label pairs, five labels, batch size 32, constant Adam at 1e-5,
   initialization seed 5 and training seed 0 all stay exactly as the authors set
   them. This protocol changes *only what is measured*.
2. **Both assignment splits come from the unused pool**, class rows 50–1522.
   The pool is permuted once with `jax.random.PRNGKey(7)`; the first 100 rows
   become development validation, the next 100 the reserved final test, and both
   are stored sorted. Seed 7 is distinct from the authors' init (5), train (0),
   eval (1) and label-pair (20) seeds, so it cannot collide with any of their
   streams.
3. **Disjointness is checked, not assumed.** `select_classes()` asserts that dev
   and final test are disjoint from each other, from the 50 training classes,
   and from rows 1523–1622 (the only classes the previous trial's
   `fsl_test_class` ever scored). All assertions pass.
4. **Recorded before evaluation.** `results/assignment/class_splits.json`
   contains the full dev and final-test class ID lists, the selection rule, all
   seeds and the task settings. It was written before any model was scored.
5. **Development validation uses 1,000 fixed sequences**, drawn once with seed
   1007 and reused at every evaluation point, matching the authors' practice of
   fixing evaluation data.
6. **Same task as training.** Two symbol–label pairs plus a supported query,
   exemplar 0, burstiness 1, one distractor, and training label pairs. Only the
   character identities are new, so the measurement is generalisation to unseen
   symbols.
7. **The final test is reserved.** Its class IDs and recipe are recorded, but no
   sequences are generated and no metric is computed. `--build-final-test`
   regenerates the data deterministically when the project is ready.
8. **`fsl_train` is retained** and is labelled *training-distribution
   evaluation*. It draws fresh sequences from the training distribution; with
   only 78,400 possible sequences under these settings it is not guaranteed
   disjoint from training, so it is a diagnostic, not validation.
9. **`fsl_test_class` is never run in assignment mode**, and is the only
   evaluator removed. `fsl_val_rl` (held-out label pairs) and `fsl_train_valex`
   (held-out exemplars) are kept because they use training classes and remain
   informative. Reproduction mode still runs all four.
10. **Training is provably unaffected** — verified below.

### Implementation

No upstream file was edited. The protocol uses `--load_eval_data`, which
`main.py` already supports (:357-369) for injecting evaluators from an HDF5
file, together with a shortened `--pe_names` list.

`make_assignment_evaluators.py` reuses the authors' own sampler functions with
our class array; `get_constant_burst_seq_idxs` accepts any sequence of class
indices, so no contiguity assumption is broken. `run_baseline.py`'s
`assignment_arguments()` removes the `fsl_test_class` entry from the five
parallel `--pe_*` lists and appends `--load_eval_data`. It asserts the upstream
evaluator lists are exactly what we expect before editing them, so a future
upstream change fails loudly rather than silently mis-aligning the lists.

In assignment mode the run therefore evaluates `fsl_dev_class` (ours, loaded
from file), `fsl_train`, `fsl_val_rl` and `fsl_train_valex`.

### Verification

Two 3,200-sequence (100-update) runs, one per protocol:

| Check | Result |
| --- | --- |
| Evaluators actually run (reproduction) | `fsl_train`, `fsl_val_rl`, `fsl_train_valex`, `fsl_test_class` |
| Evaluators actually run (assignment) | `fsl_train`, `fsl_val_rl`, `fsl_train_valex`, `fsl_dev_class` — no test-class evaluator |
| `train_loss` over 100 updates | **identical** arrays |
| `train_grad_norm` over 100 updates | **identical** arrays |
| Initial checkpoint SHA-256 | identical (`93a05460f3b0ff60…`) |
| Final checkpoint SHA-256 | identical (`2bcf5bcf2e6cf90f…`) |
| Report 01 metrics reproduced | Yes — all four evaluators matched to the quoted decimals |

Identical checkpoint hashes show that, **for this pair of 3,200-sequence runs**,
the evaluation change did not touch initialization, sampling or the training
random-number stream. The mechanism supports the general case: `main.py` splits
the training seeds (:347) before it reads any evaluator option (:356-400), so the
evaluator set cannot enter the training stream. But one pair of short runs is
evidence about those runs, not a proof covering every configuration. Stage 04
re-checked the same property independently at full scale: generation 1's initial
checkpoint is byte-identical to generation 0's.

Data checks on the generated development evaluator, all passing: shape
(1000, 3, 512) examples and (1000, 3) labels; exactly one support per sequence;
support label equals the target; the two context labels differ; every label pair
comes from the training split; and every symbol vector belongs to one of the
100 selected classes.

**One consequence worth recording.** `main.py` derives evaluator data seeds with
`jax.random.split(eval_data_seed, len(opts.pe_names))` (:371). Because
assignment mode passes three `pe_names` instead of four, the retained
diagnostics receive a *different fixed draw* of sequences — same distribution,
different sample. So `fsl_train` in assignment mode is not sequence-for-sequence
comparable with `fsl_train` in reproduction mode (we observed 25.9% versus
28.3% after 100 updates, which is sampling noise at n=1,000). This is a property
of the authors' seeding, not a defect, but cross-protocol comparisons must be
made distribution-to-distribution rather than run-to-run.

## Commands

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --timeout 120 -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/make_assignment_evaluators.py
.\.venv\Scripts\python.exe scripts/run_baseline.py --run-name protocol_check_reproduction
.\.venv\Scripts\python.exe scripts/run_baseline.py --protocol assignment --run-name protocol_check_assignment
```

Dry-run inspection of either protocol without training:

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py --full --protocol assignment --dry-run --run-name inspect_only
```

## Deviations from the authors

| Deviation | Reason |
| --- | --- |
| `fsl_test_class` not run in assignment mode | It scores classes we need to keep untouched; reproduction mode is unchanged |
| A new `fsl_dev_class` evaluator on 100 previously unused classes | Gives class-level generalisation measurement with a genuinely reserved final test |
| Evaluator data built by our script and loaded via `--load_eval_data` | Uses an option the authors provide; avoids patching their source |
| matplotlib and tqdm added | Required to import the authors' `visualize_runs.py` |
| `.gitignore` re-includes `results/assignment/class_splits.json` | The protocol record must be auditable |
| CPU execution, `use_wandb=False` | JAX has no native-Windows NVIDIA support; no account services are used |

No upstream source file was patched. `git status` under `upstream/` is clean.

## Reporting rule

Each stage report from here on states: the upstream files and functions used,
the project files created or modified, what the code does, the commands run, the
results with their verification, and the version-control and ignore status of
everything produced — including which evidence is generated and therefore not
preserved by the repository.

## Next step

Run the full single-generation baseline under `--protocol assignment` and
analyse its learning curves and induction-circuit measures. That is report 03.

> **Note added in stage 04.** Paths and script names in this report are
> historical and were deliberately left unchanged. Current locations are listed
> in `reports/04_project_cleanup_and_first_successor.md`, which also records the
> wording corrected above. No numerical result in this report changed.
