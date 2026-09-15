# 01 — Setup and single-generation baseline trial

Date: 15 September 2026. Scope: the unmodified induction-head baseline from
Singh et al. (2024), not the transience or coopetition experiments.

## Outcome

**Setup and the short CPU trial succeeded.** The authors' JAX/Equinox code loaded
the supplied data, computed finite losses and gradients, performed 100 optimizer
updates, evaluated all four original evaluators, and saved checkpoints. The final
checkpoint reloaded successfully and reproduced every saved final evaluation
metric per example within `rtol=1e-5, atol=1e-6`.

The trial took **16.33 seconds including imports, compilation and file output**.
No source patches were required. No full training, recursive training, extended
task or mechanistic analysis was run. These results establish an executable
baseline, not reproduction of the paper or formation of induction heads.

No blocker remains for CPU baseline execution. Native Windows GPU execution is
unavailable through supported JAX NVIDIA wheels. A distinct validation/final-test
protocol remains a research-design decision, described below.

## Repository and environment

- Upstream: [aadityasingh/icl-dynamics](https://github.com/aadityasingh/icl-dynamics).
- Commit: `85b895c720844e795734b9391b32d2619065f9a1`.
- Initial and final `git status --short`: empty; final `git diff --stat`: empty.
- No applicable `AGENTS.md` was found in the workspace, repository, or ancestor
  directories. Read the README's induction-head and installation sections,
  `setup.md`, `ih_paper_runs.sh`, `main.py`, `main_utils.py`, `models.py`,
  `samplers.py`, `opto.py`, and the feature extraction code.
- No Git repository was initialized, moved, committed or pushed; no remotes or
  persistent Git configuration were changed. Git's sandbox ownership warning was
  handled with a command-local `-c safe.directory=...` for this specific checkout.
  Git also warned that the user-level ignore file was inaccessible; tracked-file
  inspection and repository status completed.
- Native Windows x86-64, build `10.0.26200`, release `25H2`; Python reports
  `Windows-10-10.0.26200-SP0` and the registry retains the legacy product string
  `Windows 10 Home Single Language`. These are the observed API strings, rather
  than an inferred marketing edition.
- CPU: Intel Core i5-11400H @ 2.70 GHz; 12 logical processors visible to Python.
- NVIDIA GTX 1650, 4,096 MiB; driver 592.82; `nvidia-smi` reports CUDA capability
  13.1. This driver report does not establish an installed CUDA toolkit.
- `python` works: CPython 3.10.11, 64-bit, originally at
  `C:\Users\13327\AppData\Local\Programs\Python\Python310\python.exe`.
  `py -0p` found no registered Python installation in this sandbox. `uv` and
  `conda` were not on PATH; `python3` resolves to a WindowsApps alias.
- Used Python's built-in `venv` at `.venv/`. JAX reported `TFRT_CPU_0`;
  `jax_enable_x64=False` (float32 computation). Hardware queries through CIM
  were denied, so the report uses registry, Python and `nvidia-smi` observations.

The authors explicitly tested JAX 0.4.26 (`setup.md`). We pinned compatible
Equinox/Optax/Chex releases and NumPy 1.x rather than upgrading the old source.
The installed versions are:

| Package | Version | Package | Version |
| --- | --- | --- | --- |
| jax / jaxlib | 0.4.26 / 0.4.26 | equinox | 0.11.4 |
| optax | 0.2.2 | chex | 0.1.86 |
| numpy | 1.26.4 | scipy | 1.12.0 |
| h5py | 3.11.0 | nptyping | 2.5.0 |
| jaxtyping | 0.2.28 | ml-dtypes | 0.3.2 |
| absl-py | 2.5.0 | opt_einsum | 3.4.0 |
| toolz | 1.1.0 | typeguard | 2.13.3 |
| typing_extensions | 4.16.0 | pip / setuptools | 23.0.1 / 65.5.0 |

`requirements-lock.txt` pins all runtime packages; bootstrap pip/setuptools are
recorded separately. `pip check` reported **No broken requirements found**.
NumPy 1.26 also avoids NumPy 2 changes affecting older typing/dependency code.
The authors' `jax.tree_map`/`jax.tree_leaves` deprecation warnings remain visible
in the log; they do not prevent this pinned environment from running.

CPU was selected because the [official JAX platform table](https://docs.jax.dev/en/latest/installation.html#supported-platforms)
does not support NVIDIA GPU execution on native Windows. WSL2/Linux GPU setup
was not needed for this task. `use_wandb=False`; the launcher also sets
`WANDB_MODE=disabled`. No account login or paid service was used. PyTorch and
plotting dependencies were unnecessary for training on the supplied features.

## Actual data and task

File: `upstream/icl-dynamics/omniglot_resnet18_randomized_order_s0.h5`,
16,638,016 bytes. SHA-256:
`a8b8fa1f25d62e077b658c590f77e24447894fa325d97651dba22c6621e62b5b`.

| HDF5 path | Kind / shape | dtype |
| --- | --- | --- |
| `class_order` | dataset `(1623,)` | int64 |
| `resnet18` | group | — |
| `resnet18/224` | group | — |
| `resnet18/224/feat` | dataset `(1623, 5, 512)` | float32 |

Both datasets are uncompressed; visited objects have no attributes. `class_order`
is a permutation of 0–1622. The baseline reads the feature array directly: its
class IDs mean **row positions in this already reordered file**, not output
labels. Each row contains five fixed 512-dimensional Omniglot feature vectors.
All features are finite, with observed range 0 to 11.4714155. They are precomputed
ResNet18 features, not images or learned embeddings from this trial. The README
describes their origin; the extraction code uses an ImageNet-pretrained ResNet18.

For each training sequence, `samplers.py`:

1. Chooses a query class uniformly among rows 0–49 (`zipf_alpha=0`).
2. Chooses a different distractor class uniformly from the remaining classes.
3. Puts one query-class support and one distractor into the context, in random
   order (`burstiness=1`, distractor enabled, context length 2).
4. Samples exemplars from the allowed split. Training allows only exemplar 0,
   so the query vector exactly repeats its support vector.
5. Chooses an allowed unordered label pair, then randomly assigns its two labels
   to support/query and distractor. Both occurrences of the query class receive
   the same label. There is no stable class-to-label mapping across sequences.

Training batches have `examples: float32[32,3,512]` and `labels: int32[32,3]`.
The model interleaves these into five tokens:
`symbol A, label A, symbol B, label B, query symbol`. It excludes the query label
from the input and predicts one of five labels (0–4); only query cross-entropy
contributes to the loss. Changing the supplied target label was verified to
leave forward outputs unchanged.

Actual examples from the first training batch (all exemplar IDs are 0):

| Context | Query | Correct label | Why |
| --- | --- | --- | --- |
| class 13 → 2; class 44 → 4 | class 13 | 2 | Context assigns class 13 label 2. |
| class 33 → 0; class 43 → 1 | class 33 | 0 | Context assigns class 33 label 0. |
| class 17 → 4; class 28 → 3 | class 17 | 4 | Context assigns class 17 label 4. |

Holding the first example's class sequence `[13,44,13]` fixed and resampling
labels with key 123 gave `[4,2,4]` instead of `[2,4,2]`. This is a direct check
of reassignment, not a second training run. The inspection also checked exact
HDF5 indexing, a single matching support, different context labels, allowed label
pairs and agreement between support and query targets.

## Original baseline configuration

Selected the **first `python main.py` command**, Figure 3a's `omniglot50_rl5`
run in `ih_paper_runs.sh`; did not execute that shell script. Defaults inherited
from the Python code are included below and recorded in
`results/inspection/baseline_options.json`.

| Setting | Original value / meaning |
| --- | --- |
| Architecture | 2 causal transformer layers, 8 attention heads per layer; 64-dimensional residual stream, 8 dimensions per head |
| Feed-forward blocks | None (`mlp_ratio=None`): attention-only transformer with residual connections |
| Input/output | Learned 512→64 symbol projection with bias; learned five-label embedding; 64→5 output projection with bias |
| Normalization / positions | LayerNorm before attention and at final output; rotary positions (RoPE), timescale 10,000 |
| Other defaults | Dropout 0; no QKV or attention output projection bias; initialization rescale 1; no optogenetic clamps/ablations |
| Initialization | LeCun-normal linear weights; zero linear biases; label embeddings use truncated normal with scale 0.02 |
| Parameter count | 66,629, counted from the instantiated model |
| Optimizer | Adam, constant learning rate 0.00001; Optax defaults β1=0.9, β2=0.999, ε=1e-8; weight decay 0; no clipping/warmup in this optimizer branch |
| Batch | 32 sequences, one microbatch of 32 |
| Data | One sampler with mixing weight 1.0; uniform classes; no added noise; no random query-target manipulation |
| Context | Two symbol–label pairs plus query; five transformer tokens |
| Seeds | Model initialization 5; training 0; evaluation 1; label-pair split 20 |
| Duration | 1,000,000 **sequences**, equivalent to 31,250 updates, not one million optimizer steps or epochs |
| Evaluation | Every 5,000 sequences; 1,000 fixed sequences per evaluator; eval batch size 1,000 |
| Checkpoint | Every 1,000 sequences, plus final checkpoint; model, optimizer state and PRNG keys |

The code checks schedules at batch boundaries, rounding nonmultiples of 32
upward (e.g. evaluation requested at 5,000 occurs at 5,024). Initial evaluation
and checkpointing happen at zero; final evaluation/checkpointing are explicit.
Unused parser defaults such as warmup/decay steps do not change constant Adam.
`opto.make_fn_from_opts` was verified to select its default forward function.

## Splits and evaluator interpretation

Classes: train rows **0–49**, validation rows **50–1522**, test rows
**1523–1622** (`50 / 1473 / 100`). Exemplars: train **0**, validation **1–4**,
test **empty** (`1 / 4 / 0`). Label pairs: 8 training, 2 validation, 0 test.
With split seed 20, training pairs are `{1,2}`, `{2,3}`, `{0,4}`, `{0,3}`,
`{0,1}`, `{2,4}`, `{1,4}`, `{3,4}`; validation pairs are `{1,3}`, `{0,2}`.
Labels are individually familiar; the held-out property is their pairing.

| Evaluator | Classes | Exemplars | Label pairs | Measures |
| --- | --- | --- | --- | --- |
| `fsl_train` | Train | 0 | Train | Fresh sequences from the training distribution |
| `fsl_val_rl` | Train | 0 | Validation | Generalization to unseen label combinations |
| `fsl_train_valex` | Train | 1–4 | Train | Generalization to held-out exemplars of familiar classes |
| `fsl_test_class` | Test | 0 | Train | Generalization to 100 unseen classes |

All four have the same two-pair, supported-query task. Evaluation data are drawn
once with evaluation seed 1 and reused throughout the run. In `fsl_train_valex`,
support/query exemplars are sampled independently; they can coincide with
probability 1/4. The `match_query_and_distractors` flag is false.

**Code/documentation discrepancy:** the `pe_exemplars` help string suggests
nontraining classes automatically use all exemplars, but `main.py` directly uses
the requested exemplar split. Thus `fsl_test_class` actually uses exemplar 0.
The large 1,473-class validation partition is not used by any of these evaluators.

`acc` is five-label argmax accuracy (uniform guessing: 20%). `in_context_acc`
restricts predictions to the two labels actually present in context (guessing:
50%). `prob` is probability assigned to the target; `use_context_prob` sums
probability on both context labels. The corresponding context-restricted
probability is also logged. Out-of-context accuracy/probability are necessarily
zero here because the correct label always appears in context.

### What the assignment will need

The authors' diagnostic evaluators do not by themselves enforce an untouched
final test. `fsl_test_class` is inspected throughout training, including this
trial. `fsl_train` also shares the training distribution; distinct RNG seeds do
not guarantee disjoint examples in a finite synthetic task (there are only
78,400 possible training sequences under these settings).

Before assignment model selection, decide whether disjointness means classes,
exemplars, label pairs, full sequences, or several of these. Freeze validation
data and selection rules, reserve a separate final-test set, and evaluate that
set only after choices are fixed. If the assignment requires untouched final
classes, reserve new classes from the currently unused pool rather than treating
these already-inspected 100 classes as untouched. The unused validation classes
provide room for such a design, but no repartitioning was performed here. Keep
an authors' reproduction configuration separate from any later assignment split
configuration so changes are explicit.

## Trial differences, results and verification

The launcher changes only duration/frequencies and output bookkeeping:

| Item | Original | Trial |
| --- | --- | --- |
| Training sequences | 1,000,000 | 3,200 (100 updates) |
| Evaluate every | 5,000 | 1,600; actual counts 0, 1,600, 3,200 |
| Checkpoint every | 1,000 | 3,200; actual checkpoints 0 and 3,200 |
| Saved evaluator data | Not requested | `eval_data.h5`, for checkpoint replay |
| Output | Authors' reproduction folder/name | `results/baseline_trial_01/` |
| Runtime | Authors describe GPU setups and CPU support | Explicit CPU, local logs |

Evaluation sample counts and batch sizes were **not reduced**. Scientific
settings, data, seeds, model, optimizer and forward/loss implementation were
unchanged. No compatibility diff exists because no upstream source was edited.

Each accuracy below is over 1,000 fixed sequences; loss is mean query
cross-entropy in nats.

| Evaluator | Initial loss | Final loss | Initial accuracy | Final accuracy | Final context-restricted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fsl_train` | 1.968856 | 1.646421 | 21.5% | 28.3% | 51.2% |
| `fsl_val_rl` | 2.091902 | 1.886352 | 22.9% | 15.8% | 48.9% |
| `fsl_train_valex` | 1.924498 | 1.609357 | 21.3% | 27.5% | 52.7% |
| `fsl_test_class` | 2.028257 | 1.689733 | 19.2% | 21.7% | 47.2% |

Training loss averaged 1.906231 over the first ten batches and 1.609588 over the
last ten. Individual first/last batch losses were 1.729466 / 1.795053, illustrating
batch variability. Gradient norms were finite, ranging 5.099954–11.663898.
`grad_batch_stddev=0` is expected with one microbatch; it is not a measure of
generated-symbol diversity.

Verified final model parameters are finite and differ from initialization
(parameter-change L2 norm 0.128750). Restored Adam's update counter is 100.
All final per-example metrics matched re-evaluation after loading the checkpoint.
The restored optimizer also completed a finite extra update **in memory only**
on a verification batch; it did not alter the trial or save another model.
Target-label exclusion and five-token input shape were checked directly.

### Measured time and outputs

- Launcher wall time: **16.3333 s**, including Python startup/imports, compilation,
  data handling, three evaluations and two checkpoint writes.
- Upstream timer: **13.9984 s** (0.23330675 min), inside the training entry point;
  excludes imports. This is not pure optimizer time.
- Initial evaluation marker: 7.7188 s after launch; 1,600-sequence marker:
  15.1786 s; training-end marker: 15.8253 s. Compilation was not separately
  instrumented. The last interval, 0.6467 s, includes evaluation/logging at
  1,600 plus 50 warmed training updates and possible asynchronous dispatch.
- Checkpoint replay evaluation: **1.3150 s**, measured in a separate verification
  process, including its evaluation compilation; not an isolated inference benchmark.
- Initial: `results/baseline_trial_01/checkpoints/00000000000.eqx`.
- Final: `results/baseline_trial_01/checkpoints/00000003200.eqx`.
  Each is **809,283 bytes**, containing model, optimizer and RNG state.
- Exact config/arguments: `config.json`, `command.json`; raw metrics: `log.h5`;
  fixed trial evaluators: `eval_data.h5`; execution: `console.log`, `timing.json`;
  checkpoint checks and metric summaries: `verification.json`, all in the run folder.
- Data/environment evidence: `results/inspection/inspection.json`,
  `baseline_options.json`, `console.log`.

No circuit-strength measurements or causal head ablations were performed.
Near-50% context-restricted accuracy does not demonstrate learned induction;
the decrease on held-out label-pair accuracy after only 100 updates should not
be interpreted as recursive degradation or a reproduced paper result.

## Commands and project additions

Commands below are PowerShell from the project root. Initial installation:

```powershell
python --version
python -m pip --version
py -0p
nvidia-smi
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
```

That pip command first failed with sandbox `WinError 10013` (network denied).
It was rerun with approved network access; that attempt timed out while
downloading JAXlib (16.7/46.5 MB). The successful retry was:

```powershell
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --timeout 120 -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze
.\.venv\Scripts\python.exe scripts/inspect_baseline.py
.\.venv\Scripts\python.exe scripts/run_baseline.py --run-name baseline_trial_01
.\.venv\Scripts\python.exe scripts/verify_trial.py results/baseline_trial_01
```

The inspection was rerun after adding the `class_order` permutation check, with
stdout saved to `results/inspection/console.log`. Successful `pip freeze` output
was recorded as `requirements-lock.txt`. Future identical runtime dependency
installation should use that lock file, as shown in the project README.

Repository inspection used this command-local ownership setting (substitute
`status --short` or `diff --stat` for `rev-parse HEAD` for the other checks):

```powershell
git -c safe.directory='C:/Users/13327/OneDrive/Desktop/NLP project 1/upstream/icl-dynamics' -C upstream/icl-dynamics rev-parse HEAD
```

Source inspection used `rg` and `Get-Content`; hardware fallback commands were
`Get-ItemProperty` on `HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion` and
`HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0`. Script syntax was
checked with Python's `ast.parse` before execution.

New maintained files: `README.md`, `.gitignore`, `requirements.txt`,
`requirements-lock.txt`, `scripts/run_baseline.py`, `scripts/inspect_baseline.py`,
`scripts/verify_trial.py`, and this report. Generated files are in `.venv/` and
`results/`; ignore rules exclude them and Python caches. The root is not a Git
repository; these rules are ready for future project version control. All
authors' files remain in their original checkout.

## Recommended next step and full-run command

Agree on the assignment's validation/final-test protocol before using results
to select models or settle the research question. Then run the authors' full
baseline as a clearly labeled reproduction experiment, preserve its checkpoints
and assess accuracy and induction-head progress measures with their analysis
code. That long run has **not** been started.

Proposed command:

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py --full --run-name baseline_full_is5
```

To inspect without starting:

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py --full --dry-run --run-name baseline_full_is5
```

The underlying scientific arguments are the first baseline line in
`ih_paper_runs.sh`, with `MAIN_RUN_ITERS=1000000` and `INIT_SEED=5`; the launcher
only redirects the data path and output location in full mode. Its argument
vector is printed before execution and saved as `command.json` for actual runs.

**Estimate, not measurement:** linearly scaling the observed last 50-update
interval gives about 404 seconds (6.7 minutes) for 31,250 updates at that observed
mixed workload rate. This is a very short timing sample, includes evaluation,
does not isolate JAX synchronization, and omits the full schedule's much more
frequent checkpoint writes. Sustained CPU speed and OneDrive disk behavior can
change it substantially. Treat a full run as plausibly minutes to tens of
minutes, not a promised completion time. The original checkpoint schedule saves
1,001 checkpoints, approximately **810 MB** at the observed checkpoint size,
plus logs. Measure the full run before planning multiple seeds/generations.

Attribution: [Singh et al., *What needs to go right for an induction head? A
mechanistic study of in-context learning circuits and their formation*](https://arxiv.org/abs/2404.07129).
Configuration/task claims above come from the inspected local upstream code;
metrics and data properties come from this trial, not from the paper.
