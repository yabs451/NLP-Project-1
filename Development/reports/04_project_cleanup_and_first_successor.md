# 04 — Project cleanup, corrections, and the first recursive successor

Date: 15 September 2026. Scope: reorganise the project so every file is
explainable, correct overstated claims in reports 02 and 03, build one reusable
recursive pipeline for the **base (label-only)** task, and run exactly one
successor. The extended task is not implemented, and no further generations were
run.

## Outcome

Generation 1 completed: 1,000,000 generated examples, 31,250 optimizer updates,
all verification checks passed. It answered exactly the same number of the 1,000
development questions correctly as generation 0 (**96.7%**), with induction and
previous-token scores differing by at most 0.0015 across all sixteen heads.

> **Corrected in stage 05.** "Identical" overstated this. The two models are
> measurably different — development loss differs by 0.000493 nats, two ablation
> conditions by 0.1 accuracy point, and the final weights by a measurable amount.
> Equal accuracy on a 1,000-question set means the difference is below what that
> set can resolve (standard error about 0.54 points near 97%), not that the
> models are the same. Stage 05 introduced a 10,000-question development set so
> later comparisons have a finer instrument. No number changed; the wording
> did.

The reason is measurable rather than mysterious: the parent's argmax was wrong on
**13 of 1,000,000 examples (0.0013%)**, and those 13 are all the *same* question
repeated. One generation of label-only recursion from a near-perfect parent
changes essentially nothing. That is reported as a null result; no setting was
adjusted to manufacture degradation.

One real defect was found and fixed in our own tooling (upstream provenance was
being recorded incorrectly), and one wording ambiguity in report 03 was
corrected. **No previously reported number changed.**

## Project map

```
NLP-Project-1/
├── CLAUDE.md                      working conventions (new)
├── README.md                      full reproduction guide (rewritten)
├── requirements.txt / -lock.txt   dependency pins and full freeze
├── scripts/
│   ├── common.py                  shared helpers (new)
│   ├── prepare_evaluation_data.py evaluation split + dev questions
│   ├── inspect_source_data.py     data, environment and provenance record
│   ├── verify_run.py              soundness checks for any finished run
│   └── base_task/
│       ├── train_original.py      trains generation 0
│       ├── run_recursive.py       the recursive pipeline (new)
│       └── analyse_runs.py        curves, circuit measures, ablations
├── reports/                       01, 02, 03, 04 — operational handovers
├── findings/                      01, 02 + evidence/ — the scientific record
├── upstream/icl-dynamics/         the authors' code, unmodified
└── results/
    ├── evaluation_data/           class_splits.json (tracked), eval_dev.h5
    ├── base_task/
    │   ├── generation_0_original_data_init_seed_5/
    │   └── generation_1_generated_data_init_seed_5/
    ├── protocol_checks/           stage-02 evidence
    └── inspection/                data and environment record
```

Extended-task code will go in `scripts/extended_task/` with runs under
`results/extended_task/`. Neither exists yet — no empty folders were created.

### Maintained files

| File | Purpose | Inputs | Outputs | When it runs |
| --- | --- | --- | --- | --- |
| `scripts/common.py` | Shared helpers: project paths, the authors' baseline arguments, checkpoint loading, feature loading, and rebuilding a run's evaluators. Sets `JAX_PLATFORMS=cpu` and puts upstream on `sys.path` before JAX is imported | — | — | imported by every other script |
| `scripts/prepare_evaluation_data.py` | Picks the 100 development and 100 reserved final-test classes, checks disjointness, builds the fixed development questions | feature `.h5`, the authors' baseline arguments | `results/evaluation_data/class_splits.json` (tracked), `eval_dev.h5` | once, before any training; deterministic |
| `scripts/inspect_source_data.py` | Records the data file, resolved options, environment and upstream provenance; checks the sampler produces the intended task | feature `.h5` | `results/inspection/inspection.json`, `baseline_options.json` | once per environment |
| `scripts/verify_run.py` | Confirms a finished run is sound (any generation) | a run folder | `verification.json` in that folder | after any training run |
| `scripts/base_task/train_original.py` | Launches the authors' `main.py` as a subprocess to train generation 0 | `ih_paper_runs.sh`, `--full/--protocol/--run-name/--dry-run` | run folder with `command.json`, `console.log`, `timing.json`; upstream writes `config.json`, `log.h5`, `checkpoints/` | to create generation 0 |
| `scripts/base_task/run_recursive.py` | The recursive pipeline: generate data from a parent, then train a successor. Repeats for N generations | a parent run folder, `--generations` | successor run folder with `generated_training_data.h5`, `generation_metadata.json`, `config.json`, `log.h5`, `checkpoints/`, `timing.json` | to create generation 1 and beyond |
| `scripts/base_task/analyse_runs.py` | Learning curves, per-head previous-token and induction scores across checkpoints, attention map, head ablations | a run folder + `eval_dev.h5` | `analysis/` with `analysis.json` and three `.png` | after any training run |

Key functions in the new pipeline: `question_sampler()` (rebuilds the authors'
training sampler but returns indices), `parent_predictor()` (batched argmax, no
weight updates), `check_no_target_leak()`, `generate_dataset()`,
`worked_examples()`, `train_successor()` (one pass over the saved dataset),
`run_one_generation()` and `generation_number()`.

### Upstream files and functions used

All under `upstream/icl-dynamics/`, unmodified.

| Concern | Where | What it supplies |
| --- | --- | --- |
| Model | `models.py` `SequenceClassifier` (:596), interleaving (:748-753), `AttentionBlock` (:260) | The 2-layer attention-only transformer; drops the query label when building tokens, which is why the target cannot leak; caches `attn_scores` for our measures |
| Sampling | `samplers.py` `get_constant_burst_seq_idxs` (:32), `get_mixed_seq_idxs` (:139), `get_exemplar_inds` (:167), `fewshot_relabel` (:221), `make_data_sampler` (:288) | Question construction and per-sequence label reassignment. Our `question_sampler()` calls these directly |
| Training | `main.py` `train_step` (:87), `compute_loss` (:70), `ce` (:59), `ALL_TRAIN_METRICS` (:42) | One Adam update and the metric names our log reuses |
| Evaluation | `main.py` `evaluate` (:200), `eval_step` (:132), `make_batched_fn` (:183), evaluator construction (:356-400) | Scoring, and the recipe `common.build_evaluators()` mirrors |
| Config/model build | `main_utils.py` `create_parser` (:20), `check_opts` (:119), `get_splits_from_opts` (:153), `get_model_from_opts` (:222), `get_optimizer_from_opts` (:250) | Every default we inherit, the splits, and fresh model/optimizer construction |
| Interpretation | `visualize_runs.py` `make_forward_fn` (:701), `update_prev_token_over_time` (:338), `plot_prev_token_over_time` (:355-364), `plot_attention_over_time` (:513-526); `opto.py` `make_fn_from_opts` (:150), ablation (:368-372) | Attention-returning forward pass, the two progress-measure definitions, and head ablation by zeroing value vectors |

`main.py` also defines the training-loop structure our `train_successor()`
mirrors (:468-545) and the seed chain we reproduce (:296-297, :347-348, :525).

### Why a training loop of our own

We checked for an upstream hook first. `--load_eval_data` (`main.py`:357-369)
injects **evaluators** from a file, which is what our evaluation protocol uses,
but there is no equivalent for training data: `main.py` always samples training
batches on the fly (`train_data_sampler`, :338-345, called at :530). The only
options for data are `--data_type {file,onehot}`, which choose the *feature*
source, not pre-generated sequences.

So `train_successor()` reproduces the structure of their loop and calls their
`train_step`, `evaluate`, model, optimizer and checkpoint format, replacing one
line — the sampler call — with a read from the saved dataset. It writes
`log.h5` with their dataset names and layout, so `analyse_runs.py` and their
tooling read a successor exactly like generation 0. Upstream stays untouched.

### Generated files

| Kind | Contents |
| --- | --- |
| `config.json` | every resolved option including seeds and the full evaluation/checkpoint schedules. `eval_every`/`ckpt_every` read `None` because `main.py` expands them into `eval_sched`/`ckpt_sched` and clears the originals (:239-253) — not a missing value |
| `log.h5` | `train_loss`, `train_grad_norm`, `train_grad_batch_stddev`, `train_iter` (31,250 entries each); `eval_iter` (201); and per evaluator `acc`, `loss`, `in_context_acc`, `prob`, `use_context_prob`, `out_context_*` as (201, 1000) arrays — one value per evaluation point per sequence |
| `checkpoints/` | 1,001 `.eqx` files per run, named by sequence count, 809,283 bytes each (~800 MB total). Each holds model weights, optimizer state and the three PRNG keys |
| `generated_training_data.h5` | successors only: `class_idxs` (N,3) int16, `exemplar_idxs` (N,3) int8, `labels` (N,3) int8 with the query target replaced, `true_query_label` (N,) int8 for diagnostics. Indices not feature vectors, so 1,000,000 examples occupy 4.3 MB instead of ~6 GB |
| `generation_metadata.json` | successors only: generation number, parent checkpoint identity, source-data identity, generation rule, seeds, target-quality statistics, worked examples |
| `analysis/` | `analysis.json` plus `curves.png`, `head_measures.png`, `attention_example.png` |
| `verification.json`, `timing.json` | `verify_run.py` output; measured durations |

## Files created, moved, modified and removed

| Change | Path | Reason |
| --- | --- | --- |
| Created | `CLAUDE.md` | No agent-instruction file existed; conventions were only implicit in reports |
| Created | `scripts/common.py` | `load_checkpoint` and the upstream-import preamble were duplicated across scripts |
| Created | `scripts/base_task/run_recursive.py` | The recursive pipeline |
| Created | `findings/` + `findings/evidence/` | Separates the scientific record from operational handovers |
| Created | `reports/04_...md` | This report |
| Moved | `scripts/run_baseline.py` → `scripts/base_task/train_original.py` | Groups base-task code; the name says what it trains |
| Moved | `scripts/analyse_baseline.py` → `scripts/base_task/analyse_runs.py` | It analyses any generation, not just the baseline |
| Moved | `scripts/make_assignment_evaluators.py` → `scripts/prepare_evaluation_data.py` | Shared by both tasks; "assignment" was ambiguous |
| Moved | `scripts/inspect_baseline.py` → `scripts/inspect_source_data.py` | It inspects the source data, not a run |
| Moved + rewritten | `scripts/verify_trial.py` → `scripts/verify_run.py` | The old one only worked on short trials that saved `eval_data.h5`; it now works on any run by rebuilding evaluators |
| Moved | `results/assignment/` → `results/evaluation_data/` | "assignment" read as coursework rather than class assignment |
| Moved | `results/baseline_full_is5_assignment/` → `results/base_task/generation_0_original_data_init_seed_5/` | `is5` was an unexplained abbreviation for initialisation seed 5 |
| Moved | `results/protocol_check_{reproduction,assignment}/` → `results/protocol_checks/{original,assignment}_evaluators/` | Groups stage-02 evidence |
| Modified | `.gitignore` | New `results/` layout; keeps the split record trackable |
| Modified | `README.md` | Rewritten as a reproduction guide |
| Modified | `reports/02`, `reports/03` | Corrections below, plus a note that their paths are historical |
| Modified | `scripts/inspect_source_data.py` | Fixed the upstream provenance defect below |
| **Removed** | nothing | No results, checkpoints or reports were deleted. `verify_trial.py` was rewritten in place via `git mv`, not deleted |

All moves used `git mv` where the file was tracked, so history is preserved.
Nothing imports the old names; every script compiles and was re-run.

**Filenames the authors' program produces — `config.json`, `log.h5` and the
zero-padded checkpoint names — were deliberately left alone**, so their tooling
and ours keep working.

## Corrections to earlier reports

**No numerical result changed.** Every correction is wording or scope.

| # | Where | Was | Now |
| --- | --- | --- | --- |
| 1 | 03, previous-token measure | "the 1−a … is spread over the *i* other causally allowed positions, so the corrected score is a − (1−a)/(i+1)" — the text said *i*, the formula said *i+1* | States the index `r` explicitly, gives the formula `a − (1 − a)/(1 + r)`, explains that `1 + r` is the count of causally visible positions **other than** the previous token, and works an example |
| 2 | 03, ablation | "no individual head is necessary" | "points to redundancy … does **not** prove that no individual head is necessary": four of sixteen heads, one at a time, one model |
| 3 | 03, phase 3 | "the previous-token score rises first and fastest, the induction score follows" | Reports the measurements, then says they are *consistent with* that ordering but do not establish it: checkpoints are tens of thousands of sequences apart and the measures are on different scales |
| 4 | 03, outcome | "the run shows the picture the paper describes" | "qualitatively consistent with … We did not compare against any published number, so this is not a reproduction" |
| 5 | 03, `fsl_train_valex` | "The model becomes increasingly confident on a distribution it does not fully master" stated as fact | Split into **Observation** and **Proposed explanation, not tested here**, with an explicit note that it is not model collapse |
| 6 | 02 and 03, protocol check | "verified to leave training **bit-identical**" | Scoped to the pair of 3,200-sequence runs actually tested, with the mechanism stated separately |
| 7 | 02, 03 | paths and script names | Left unchanged as historical provenance, with a note pointing at the current locations |

### The previous-token measure: wording only, code was correct

The concern was whether our implementation matched the authors'. It does.

The authors build a shortened array with `inds = arange(1, seq)`, so index `r`
means original token position `r + 1`, and subtract
`(1 - attention) / (1 + r)` (`visualize_runs.py`:362-364). Our code computes
`raw - (1 - raw) / (1 + np.arange(seq - 1))` — the same thing.

Why `1 + r` is right, for `r = 1`: that is token position 2, which can causally
attend to {0, 1, 2}. Its previous token is position 1, leaving {0, 2} — **two**
other positions, and `1 + r = 2`. Checked across all positions: with attention
uniform over everything a token can see, the corrected score is exactly 0.000000
at every index. Stage 03 had already confirmed our output equals the authors'
own `update_prev_token_over_time` to within 1e-6 on the same checkpoint.

So: **wording was ambiguous, code was correct, no analysis was regenerated and
no number changed.** The same ambiguous sentence in `analyse_runs.py`'s
docstring and an inline comment were corrected too.

### A real defect found in our tooling

`inspect_baseline.py` ran `git -C upstream/icl-dynamics rev-parse HEAD` and
recorded the answer as `upstream_commit`. Because upstream is **vendored** — the
authors' files are tracked in our repository with no repository of their own —
that command answered with *our* project's commit (`05cfb4b…`), and
`upstream_status` returned *our* repository's dirty state. The field looked like
provenance but was wrong.

Fixed in `inspect_source_data.py`: it now records `vendored: true`, the authors'
commit as the constant recorded in report 01, our own commit under a correctly
named key, `git status --short -- upstream` (currently empty, confirming the
authors' files are unmodified), the tracked file count (35) and a SHA-256 of the
vendored tree so drift is detectable. Report 02 had flagged this area as a
limitation but described it as the field being unavailable; in fact it was
producing a misleading value.

## The recursive data recipe

For each successor, exactly:

1. **Questions** — the *original* training distribution, unchanged: the 50
   training classes (rows 0–49), exemplar 0 only, context length 2, burstiness
   1, one distractor, and the 8 training label pairs. Context labels are the
   correct ones from the authors' own sampler. Development and reserved
   final-test classes are never touched.
2. **Question stream** — we reuse the baseline's own training key chain
   (`jax.random.split(PRNGKey(train_seed), 2)` then one split per batch), so the
   successor sees **the same questions in the same order** as its parent's
   training stream. Verified: our reconstruction reproduces the authors' jitted
   sampler's `examples` and `labels` exactly for the batches checked, and batch 0
   matches the examples recorded in report 01 (classes 13→2, 44→4, query 13).
3. **Targets** — the parent's final checkpoint is run with no weight updates,
   and its **argmax over all five output labels** replaces the query's target.
   Nothing is sampled, temperature-adjusted, restricted to context labels,
   filtered by correctness, or mixed with true answers.
4. **What is preserved** — only `labels[:, -1]` changes. Context labels and input
   symbols are untouched (asserted). The target cannot reach the input: the
   authors' model drops the query label when interleaving
   (`models.py`:752), and `check_no_target_leak()` confirms that changing the
   target leaves the parent's logits bit-identical.
5. **Storage** — class/exemplar/label indices, not feature vectors, so
   1,000,000 examples are 4.3 MB. Inference runs in batches of 2,000. True
   answers are stored separately as `true_query_label` for diagnostics and are
   **never** used as a training target.
6. **Duplicates are expected.** 78,400 distinct questions appeared across the
   1,000,000 examples, about 12.8 occurrences each. Repetition follows from
   sampling a million times from a finite question space; *(correction: this
   originally read "the task has only 78,400 possible questions". That number is
   a count of what appeared in this dataset, not the size of the space, which we
   never established.)*
7. **Training** — fresh initialisation from seed 5 with a fresh optimizer, *not*
   the parent's weights; same architecture, optimizer, learning rate and batch
   size; exactly one pass over the million examples = 31,250 updates. No tuning,
   no early stopping.

### Comparability with generation 0

Generation 0 sampled its questions during training; generation 1 reads a saved
dataset. Because we reproduced the same key chain, the questions and their order
are identical, so this difference has **no practical effect here**. Two residual
differences are worth stating:

- Generation 1's training loss is computed against the **parent's generated
  targets**; generation 0's is against true answers. The two training-loss
  curves are not measuring the same thing. Development loss is against true
  answers for both, and is the comparable quantity.
- Generation 1 is **not statistically independent** of generation 0: same
  questions, same order, same initialisation, same model seed. This maximises
  control — any difference is attributable to the changed targets alone — but it
  means the agreement between them is partly by construction.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py
.\.venv\Scripts\python.exe scripts/base_task/run_recursive.py --parent results/base_task/generation_0_original_data_init_seed_5 --generations 1
.\.venv\Scripts\python.exe scripts/verify_run.py results/base_task/generation_1_generated_data_init_seed_5
.\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py results/base_task/generation_1_generated_data_init_seed_5
```

Generation numbering: generation 0 is the original model; generation 1 is the
first successor. `--generations N` trains N **additional** models, so after
`--generations 1` there are 2 models in total.

## The worked example

Both examples are real rows of `generated_training_data.h5`.

**Typical (index 0).** Context: class 13 → label 2, class 44 → label 4. Query:
class 13, exemplar 0. True answer: **2**. Parent probabilities over the five
labels: `[0.000075, 0.000009, 0.999805, 0.000025, 0.000086]`. Argmax **2**,
stored as the training target. Correct.

**A genuine error (index 5257).** Context: class 11 → label 0, class 18 → label
4. Query: class 11, exemplar 0. True answer: **0**. Parent probabilities:
`[0.221227, 0.000037, 0.000756, 0.000058, 0.777922]`. Argmax **4** — the
*distractor's* label — stored as the training target. The parent is confidently
wrong and copies the wrong context item.

## Generated-target quality

| Measure | Value |
| --- | ---: |
| Examples generated | 1,000,000 |
| Parent argmax error rate vs true answers | **0.0013%** (13) |
| Parent predicted a label absent from context | **0.0000%** (0) |
| Distinct questions appearing in the dataset | 78,400 |
| Distinct questions mislabelled | **1** |
| Generated label distribution | 188438 / 187307 / 187256 / 187105 / 249894 |
| True label distribution | 188451 / 187307 / 187256 / 187105 / 249881 |

All 13 errors are the same question — classes [11, 18, 11] with context labels
{0, 4} — which appears 13 times and is mislabelled every time. The parent is
deterministic, so its errors are systematic, not noise.

This is measured on the **actual generated training questions**. It is a
different quantity from the parent's 96.7% development accuracy on unseen
classes, and the parent's 100% `fsl_train` diagnostic does not by itself show the
targets are right — the diagnostic is 1,000 sampled sequences, while this covers
the 78,400 distinct questions that appeared in this dataset.

## Successor results

| Measure | Generation 0 | Generation 1 |
| --- | ---: | ---: |
| Dev accuracy, 100 unseen classes | 96.70% | **96.70%** |
| Dev context-restricted accuracy | 96.70% | 96.70% |
| Dev loss (true answers) | 0.0828 | 0.0833 |
| `fsl_train` | 100.0% | 100.0% |
| `fsl_val_rl` | 99.6% | 99.6% |
| `fsl_train_valex` | 86.2% | 86.2% |
| Parameter change from init (L2) | 3.7535 | 3.7543 |
| Strongest induction score (L1H3) | +0.6859 | +0.6856 |
| Strongest prev-token score (L0H2) | +0.8784 | +0.8786 |

Largest difference across all sixteen heads: 0.0015 (induction), 0.0010
(previous-token).

**Head selection.** Candidate heads were chosen from **each model's own final
scores**, never inherited. Independently, both models selected the same four
heads (L1H3, L1H6, L0H2, L0H0). That agreement is a result in itself and is what
makes the ablation comparison meaningful — we did not assume a head keeps its
function across models.

| Ablated (value vectors zeroed) | Gen 0 | Gen 1 |
| --- | ---: | ---: |
| nothing | 96.7% | 96.7% |
| L1H3, strongest induction | 92.8% | 92.8% |
| L1H6, weakest induction | 96.6% | 96.7% |
| L0H2, strongest previous-token | 88.9% | 88.8% |
| L0H0, most negative previous-token | 93.1% | 93.1% |

**Interpretation.** One generation of label-only recursion from a parent that is
essentially perfect on the training distribution produces no change this
evaluator can resolve. (The differences that do exist are listed in the
correction note above; they are real but tiny.)
The entire difference between the two training sets is one corrupted question out
of the 78,400 distinct questions in the dataset. This shows the pipeline works and
gives a clean baseline;
it says nothing yet about whether circuit function degrades before accuracy,
which needs a longer chain of generations. We did not change any setting to try
to produce collapse.

## Timings, storage and checks

| Step | Time | Storage |
| --- | ---: | ---: |
| Data generation (1,000,000 examples) | **27.9 s** | 4.3 MB dataset |
| Successor training (31,250 updates) | **462.1 s = 7.7 min** | 810 MB checkpoints |
| Verification | 10 s | 4 KB |
| Analysis | 11 s | ~400 KB |
| **Per additional generation** | **~8.2 min** | **~810 MB** |

All timings are approximate and vary by a minute or two between runs depending
on what else the machine is doing. Generation 0's training took 5.6 minutes via
the authors' `main.py`, but a stage-05 candidate doing identical work through
the same path took 7.7 minutes, so the gap between our loop and theirs is
smaller than the raw figures suggest and should not be read as a precise
measurement of our loop's overhead.
Disk after both generations: 56 GB free. **At ~810 MB per generation, about 10
more generations fit comfortably; beyond that the checkpoint schedule should be
thinned.**

Checks performed, all passing:

| Check | Result |
| --- | --- |
| Saved dataset holds the intended questions | classes all within the 50 training rows; every exemplar 0; labels within 0–4; exactly one support per sequence; context labels distinct |
| Only the query target was replaced | `labels[:, :-1]` asserted identical to the pre-replacement array |
| Target cannot leak into the input | changing the target leaves the parent's logits bit-identical |
| Question stream matches the authors' sampler | reconstructed `examples` and `labels` identical to their jitted sampler; batch 0 matches report 01 |
| Successor starts from the intended fresh initialisation | generation 1's checkpoint 0 is byte-identical to generation 0's |
| Update count | 31,250 restored from the optimizer state, matching 1,000,000 / 32 |
| Final checkpoint loads | yes, `iter` = 1,000,000; 66,629 finite parameters |
| Training log complete | 31,250 losses and gradient norms, all finite |
| Re-scoring reproduces the log | all four evaluators match to 1e-6 |
| Development evaluation uses the established questions | `build_evaluators()` rebuilt from the run's own seeds reproduces generation 0's logged per-example metrics exactly |
| Final test untouched | `eval_final_test.h5` does not exist; no evaluator named `final_test` appears in either log |
| Paths after reorganisation | all seven scripts compile and were re-run; both protocols produce the right evaluator lists; `prepare_evaluation_data.py` regenerates **identical class IDs, seeds and rule** (only the recorded folder path changed) |

## Limitations

- **Stability may simply reflect a near-perfect parent.** With 13 errors in a
  million there is almost nothing to propagate. Degradation may require many
  generations, a deliberately weaker parent, or the extended task.
- **Two generations cannot establish a temporal claim** about circuit
  deterioration preceding performance decline.
- **One seed** (initialisation 5) throughout.
- **Generation 1 is not statistically independent** of generation 0, by design.
- **Ablation remains single-head, zero-ablation, four of sixteen heads.**
- **Unresolved:** why the parent fails on classes 11 vs 18 with context labels
  {0, 4}. The obvious first check — whether those two feature vectors are
  unusually similar — was not done.
- **Unresolved:** the `fsl_train_valex` loss rise, which reappears identically in
  generation 1 (0.5933 → 0.5943).
- **Hyperparameter tuning is deferred** to a later stage, after this pilot and
  before the main controlled comparison. Any change to training settings requires
  retraining the baseline to match; settings must not drift between generations.

## Findings

- `findings/01_baseline_induction_circuit.md` — generation 0 (backfilled)
- `findings/02_first_recursive_generation.md` — generation 1
- `findings/evidence/` — `generation_{0,1}_analysis.json`,
  `generation_1_data_generation.json`, and the curve, head-measure and
  attention figures (776 KB total, tracked)

## Git status

**Nothing was committed, pushed, reset, or discarded by this stage.** No remote
was changed.

> **Updated in stage 05.** The user has since committed this stage's work as
> `ed6d2ba` ("third commit for NLP Project 1 after training generation 1 of
> baseline") and pushed it: `git ls-remote origin` now reports `refs/heads/main`
> at `ed6d2ba`. So everything listed below **is** on GitHub now, including all
> four reports, both findings and the findings evidence. The status below
> records the working tree as this stage left it.

At the time of writing, HEAD was `05cfb4b` ("Second Commit for NLP Project 1
after training baseline"), matching the remote, and everything below was
**uncommitted working-tree change, present only on this machine**:

```
 M .gitignore
 M README.md
 M reports/02_code_structure_and_validation.md
 M reports/03_full_baseline_training_and_analysis.md
RM results/assignment/class_splits.json -> results/evaluation_data/class_splits.json
RM scripts/analyse_baseline.py           -> scripts/base_task/analyse_runs.py
RM scripts/run_baseline.py               -> scripts/base_task/train_original.py
RM scripts/inspect_baseline.py           -> scripts/inspect_source_data.py
RM scripts/make_assignment_evaluators.py -> scripts/prepare_evaluation_data.py
RM scripts/verify_trial.py               -> scripts/verify_run.py
?? CLAUDE.md
?? findings/
?? scripts/base_task/run_recursive.py
?? scripts/common.py
```

`R` = renamed (staged by `git mv`), `M` = modified, `??` = untracked. Nothing is
committed and nothing is pushed, so **nothing in this stage exists on GitHub.**

Ignored and therefore not in Git: `.venv/`, caches, and everything under
`results/` except `class_splits.json` — including both 800 MB run folders, the
generated dataset and `eval_dev.h5`. A fresh clone gets the code, the authors'
code, the feature file, the evaluation protocol record, and all reports and
findings with their numbers and figures (the three `.png` figures and the
`analysis.json` files in `findings/evidence/` are tracked, so the figures do
survive a clone); the environment and every run must be regenerated. Ignored
files are not backed up anywhere.

## Next step

Run more generations with the existing pipeline — nothing needs editing:

```powershell
.\.venv\Scripts\python.exe scripts/base_task/run_recursive.py --parent results/base_task/generation_1_generated_data_init_seed_5 --generations 3
```

That would train generations 2, 3 and 4 in sequence at ~8.2 minutes and ~810 MB
each. **This was not run.** Given how stable generation 1 is, the more
informative variants to consider first are a deliberately weaker parent (fewer
training sequences, so more errors to propagate) or moving to the extended task,
where generated *symbols* introduce real distributional drift. Both are
decisions for the supervisor rather than routine implementation choices.

Attribution: [Singh et al., *What needs to go right for an induction head?*](https://arxiv.org/abs/2404.07129).
Task, model, samplers, training update, evaluation and progress-measure
definitions are theirs; the recursive pipeline, evaluation protocol and analysis
wrapper are ours. All numbers here come from the runs described.
