# Working conventions for this project

Honours NLP research project on the interpretability of model collapse, built on
Singh et al. (2024) and their vendored implementation in `upstream/icl-dynamics/`.
These conventions apply to every agent and every person working here.

## Minimal, student-readable code comes first

This is assessed coursework. The student must be able to understand and explain
every maintained script and every scientific output. Code that works but cannot
be explained in a supervision is not finished work.

**Before adding or keeping anything, answer this:** *what current, explicitly
requested experiment, result, figure, interpretation or reproduction step
requires this?* If there is no answer, do not add it and do not keep it.

Never justify a file, option or output by future usefulness, generic
robustness, convenience, or making an agent's job easier.

- Descriptive names for files, folders, functions and variables.
- Straightforward functions with meaningful jobs — a sensible middle ground
  between one enormous function and a swarm of tiny helpers.
- Concise comments guiding the reader through important steps, including inside
  `main()`. Comment the *why*, especially scientific choices.
- Brief function docstrings where inputs, outputs, shapes or unfamiliar
  operations need explaining. `[batch, 3, 512]` and "loss in nats" save a reader
  ten minutes.
- **No long docstrings or methodology essays at the top of a Python file.** A
  short summary and a pointer to the relevant Markdown.
- No unnecessary classes, wrappers, registries, frameworks or speculative
  abstractions.
- Reuse and adapt existing scripts. Never write a separate implementation per
  experiment variant, and never duplicate an evaluation workflow.

A file may have been essential earlier and become obsolete after a change of
direction. Remove retired scripts, options and outputs rather than accumulating
old versions — but check what depends on something before removing it, and
preserve evidence the research still uses. If a purpose is unclear, find the
specific dependency instead of deleting blindly or inventing a reason to keep it.

## Folder responsibilities

- `scripts/` — maintained experiment and analysis code.
- `results/` — numerical outputs, necessary settings, saved models, comparison
  tables and generated figures.
- `findings/` — written scientific interpretation, referencing the real files
  under `results/`.
- `Development/` — local, Git-ignored operational and debugging material:
  `Development/reports/` for operational reports and handovers,
  `Development/outputs/` for temporary debugging output.
- `temporary_checks/` — retired functionality kept at the user's request. A
  deliberate, currently **trackable** exception: do not add it to `.gitignore`.
  Code under `scripts/` must never import from it, and no reproduction step may
  depend on it. It holds only the retired 10,000-question evaluator rebuild.
- `upstream/icl-dynamics/` — the authors' code, vendored unmodified.

**Do not copy measurements or figures into `findings/`.** A finding references
the file under `results/` where the measurement actually lives. One copy only.

`findings/` tracking status is the user's decision and has not been made. Do not
change it, and never let code under `scripts/` depend on `findings/`.

`Development/` is not a dumping ground. Development-only output is off by
default, the main experiment must never depend on it, and reproduction must
never require a temporary check. Keep material there only while there is a
current need for it.

## Leave upstream alone

Reuse the authors' model, sampler, optimizer, loss, update, evaluation and
checkpoint functions rather than reimplementing them. If something cannot be
done through their options, write a small adapter in our scripts and document
why. Any unavoidable patch must be called out explicitly in a report. Leave
their attribution unchanged.

## Checks that protect the experiment

Keep only these safeguards: no data leakage (the query target must never reach
the model input), no invalid inputs or training (finite losses, the expected
update count, the requested configuration actually reaching the run), and no
accidental overwriting or silent reuse (refuse to write into an existing run
folder; refuse to reuse a run whose configuration does not match).

Do not build a test suite, an audit framework or a cleanup utility. Temporary
debugging checks belong in `Development/` and are never reproduction steps.

## Outputs we do and do not keep

Keep, because an analysis or finding reads them: `config.json` (what settings
ran), `log.h5` (the numbers behind the learning curves), `checkpoints/`, the
evaluation split record, tuning results and selection records, generated
datasets, and generated figures.

Do not produce: runtime/timing files or an automatic runtime column, console
transcripts as saved artefacts, standalone verification reports, setup
inspection dumps, or metadata that duplicates `config.json`.

## Checkpoint policy

Main mechanistic runs save 55 checkpoints directly during training:
initialisation, four early snapshots near 1,000 / 2,000 / 5,000 / 10,000
sequences, then every 20,000 sequences through 1,000,000. Requests land on the
next batch boundary, so the early ones are saved at 1,024 / 2,016 / 5,024 /
10,016; record the actual counts.

Do not save a dense schedule and prune afterwards. Tuning runs keep only the
final checkpoint plus any snapshot with an identified current analysis use.
Reducing snapshots must never reduce the accuracy and loss curves in `log.h5`,
which are logged separately. Adapt analysis to the checkpoints that exist; never
fabricate a missing measurement or imply sparse snapshots preserve every state.

## Scientific writing standards

- Separate observations from proposed explanations. Say which is which.
- Distinguish attention-pattern evidence from the effect of an intervention. A
  head showing a pattern has not been shown to cause a behaviour.
- State what was actually checked, not what it suggests in general.
- Do not claim reproduction of the paper without the comparisons that would
  establish it. Do not describe ordinary training behaviour as model collapse.
- Report negative and null results honestly. Never change an experiment's
  settings to obtain a more interesting outcome.
- Correct earlier documents openly, and say whether any number changed.

## Generations

Generation 0 is the original model trained on the real task. Generation 1 is the
first successor trained on data generated by generation 0, and so on. "N
additional generations" means N successors; the total number of models is N+1.
Always make clear which of the two a number refers to.

## Two kinds of document

- `Development/reports/` — operational handovers: what changed and why,
  commands, checks, results, unresolved issues, Git status. Numbered by stage.
- `findings/` — the scientific record: the question, the setup, results with
  figures, interpretation and limitations, plus the source run and reproduction
  command. Chronological.

## Version control

- Track: code, the class split record, `README.md` / `CLAUDE.md`, and
  `temporary_checks/` (excluding any large file it regenerates).
- Ignore: environments, caches, `Development/`, and `results/` apart from the
  class split record.
- A fresh clone has no run outputs. Any number a finding relies on must be
  quoted in that finding, not only stored under `results/`.
- Do not commit, push, reset or change remotes unless asked. Never assume
  ignored files are backed up anywhere.

## Agent output and monitoring limits

These exist because an earlier session exhausted its context on avoidable output.

- **Do not print entire large logs, JSON files, datasets or directory listings
  into the conversation.** Print only what the current task needs. This limits
  unnecessary *output*, not understanding: read whatever code is needed to
  modify it correctly.
- **Do not poll running jobs.** Check once when starting and once when
  finished. More checks only if the user asks or something actually goes wrong.
- Use a completion notification where one is available. Otherwise let the job
  run and return control instead of repeatedly checking it.
- Diagnose a failure from its actual error message, not by launching broad
  speculative checks.
- Do not create monitoring files, background watchers or extra test scripts.

## Keep the README usable

The README must let someone reproduce the project from a clean machine. Update
it in the same change that moves a path, renames a script or alters a command.
