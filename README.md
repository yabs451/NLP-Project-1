# Honours NLP: induction-head baseline

Initial single-generation setup for our Honours research project, building on
[Singh et al. (2024), *What needs to go right for an induction head?*](https://arxiv.org/abs/2404.07129).
The authors' [JAX/Equinox implementation](https://github.com/aadityasingh/icl-dynamics)
is preserved in `upstream/icl-dynamics/` at commit
`85b895c720844e795734b9391b32d2619065f9a1`.
Their repository covers several papers; our launcher selects only the first
unmodified baseline in `ih_paper_runs.sh`.

Completed: pinned CPU environment, data inspection and a 100-update trial in
16.33 seconds, with successful checkpoint reload and evaluation replay.

## Setup (PowerShell, from this folder)

Use Python 3.10 (this machine has 3.10.11). No environment activation is needed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --timeout 120 -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/inspect_baseline.py
```

`requirements.txt` contains the chosen compatibility pins; `requirements-lock.txt`
records all installed runtime dependencies. JAX/JAXlib stay at the authors' tested
0.4.26. The launchers select CPU and local HDF5 logs. Native Windows NVIDIA GPU
execution is unsupported by JAX ([platform table](https://docs.jax.dev/en/latest/installation.html#supported-platforms)).
The bundled feature file is sufficient: no PyTorch, image downloads or tracking
account is needed.

## Short trial

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py --run-name baseline_trial_01
.\.venv\Scripts\python.exe scripts/verify_trial.py results/baseline_trial_01
```

Each run must have a new name; omit `--run-name` for a timestamped folder. The
launcher refuses to overwrite an existing run. The trial processes 3,200 sequences
(100 updates), evaluates at 0/1,600/3,200, and saves initial/final checkpoints.
Model, optimizer, batch size, data splits and all four 1,000-sequence evaluators
retain the authors' settings. This is an execution check, not paper reproduction
or evidence that induction heads have formed.

## Full baseline (proposed; not started)

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py --full --run-name baseline_full_is5
```

Add `--dry-run` to display the underlying command without training. Full mode
uses the original 1,000,000-sequence schedule (31,250 updates), evaluation every
5,000 sequences and checkpointing every 1,000. See the report before adopting
the original evaluators for assignment model selection: its test evaluator is
used repeatedly during training.

## Files and status

- `scripts/run_baseline.py`: extracts the baseline command and launches upstream `main.py`; saves command, console log and timing.
- `scripts/inspect_baseline.py`: inspects the actual HDF5 file, resolves defaults, records environment versions and checks readable sampler examples.
- `scripts/verify_trial.py`: reloads initial/final checkpoints, verifies parameter and optimizer updates, replays saved evaluations and checks that the query label is excluded from model inputs.
- `results/inspection/`: generated environment/data inspection and baseline options.
- `results/<run>/`: generated config, metrics (`log.h5`), checkpoints, saved trial evaluators, console log, timing and verification.
- [reports/01_setup_and_baseline_trial.md](reports/01_setup_and_baseline_trial.md): completed work, measured results, compatibility decisions and split observations.
- `requirements.txt`, `requirements-lock.txt`: compatibility pins and installed dependency lock.
- `.gitignore`: excludes environments, caches and generated results from a future project-level Git repository. No repository was initialized or rearranged.

Planned work: define distinct training/validation/final-test use, conduct full
baseline reproduction, then design recursive generations and the extended task.
Recursive training and extended predictions are not implemented. The final
research question remains open.
