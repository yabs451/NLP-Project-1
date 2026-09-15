# Honours NLP: interpretability of model collapse

Research project building on
[Singh et al. (2024), *What needs to go right for an induction head?*](https://arxiv.org/abs/2404.07129).
The authors' [JAX/Equinox implementation](https://github.com/aadityasingh/icl-dynamics)
is vendored unmodified in `upstream/icl-dynamics/` at commit
`85b895c720844e795734b9391b32d2619065f9a1`. Their repository covers several
papers; our launcher selects only the first baseline in `ih_paper_runs.sh`.

**Status.** One full single-generation baseline is trained and analysed. It
reaches **96.7% development accuracy on 100 unseen character classes** in 5.59
minutes on CPU, with previous-token and induction heads forming in a sharp
transition around 150k–400k training sequences. Recursive training, the extended
task and data-mixture experiments are not implemented yet, and the final
research question is still open.

## Setup (PowerShell, from this folder)

Use Python 3.10 (this machine has 3.10.11). No environment activation is needed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --timeout 120 -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` holds the compatibility pins we chose; `requirements-lock.txt`
records everything installed. JAX/jaxlib stay at the authors' tested 0.4.26;
matplotlib and tqdm are needed only to import the authors' `visualize_runs.py`
for analysis. Native Windows NVIDIA GPU execution is unsupported by JAX
([platform table](https://docs.jax.dev/en/latest/installation.html#supported-platforms)),
so everything runs on CPU. The bundled feature file is sufficient — no PyTorch,
image downloads or tracking account is needed.

## Workflow

**1. Fix the assignment splits** (once; deterministic, safe to re-run):

```powershell
.\.venv\Scripts\python.exe scripts/make_assignment_evaluators.py
```

Selects 100 development-validation classes and 100 disjoint final-test classes
from the 1,473 classes no original evaluator touches, and writes
`results/assignment/class_splits.json` (tracked in Git) plus the fixed
development sequences. The final test is **reserved**: no data is generated and
nothing is scored on it until the project is ready.

**2. Train.**

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py --full --protocol assignment --run-name baseline_full_is5_assignment
```

* `--protocol assignment` — our development evaluator; the authors'
  repeatedly-inspected test-class evaluator is **not** run.
* `--protocol reproduction` (default) — the authors' original four evaluators.
* `--full` — the original 1,000,000-sequence schedule (31,250 updates).
  Omit it for a 3,200-sequence smoke test.
* `--dry-run` — print the underlying command without training.

Protocol choice affects evaluation only: reproduction and assignment runs
produce bit-identical checkpoints. Each run needs a new name; the launcher
refuses to overwrite. Budget about **800 MB and 6 minutes** per full run.

**3. Analyse.**

```powershell
.\.venv\Scripts\python.exe scripts/analyse_baseline.py results/baseline_full_is5_assignment
```

Writes `analysis/` inside the run folder: learning curves, per-head
previous-token and induction progress measures across checkpoints, an attention
map, and final-checkpoint head ablations. Candidate heads are chosen from
observed behaviour, not hardcoded indices.

Optional checks: `scripts/inspect_baseline.py` records the data and resolved
options; `scripts/verify_trial.py <run>` replays evaluations from checkpoints
(needs a run started without `--full`, which saves its evaluator data).

## Files

- `scripts/run_baseline.py` — extracts the baseline command from `ih_paper_runs.sh` and launches upstream `main.py`; implements `--protocol`.
- `scripts/make_assignment_evaluators.py` — selects and records the validation/final-test classes; builds the development evaluator.
- `scripts/analyse_baseline.py` — curves, induction-circuit measures, attention maps, head ablations.
- `scripts/inspect_baseline.py` — inspects the HDF5 data, resolved defaults, environment and sampler output.
- `scripts/verify_trial.py` — reloads checkpoints and replays saved evaluations.
- `reports/01_setup_and_baseline_trial.md` — environment, data and the first 100-update trial.
- `reports/02_code_structure_and_validation.md` — code map, recovery, and the assignment evaluation protocol.
- `reports/03_full_baseline_training_and_analysis.md` — the full run, results and mechanistic analysis.
- `results/` — generated. Ignored by Git **except** `results/assignment/class_splits.json`, which is the protocol record and must stay auditable.

## Reporting rule

Each stage report states: the upstream files and functions used; the project
files created or modified and what they do; the commands run; the results with
their verification; and the version-control and ignore status of what was
produced — including which evidence is generated and therefore not preserved by
the repository. Because run folders are ignored, any number a report relies on
is quoted in that report.

## Next step

Implement the first recursive generation: train generation 2 on data sampled
from generation 1's predictions and compare both on the same fixed development
evaluator and the same mechanistic measures. See the end of report 03 for the
two decisions to settle first.
