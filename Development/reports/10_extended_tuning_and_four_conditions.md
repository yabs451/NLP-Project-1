# 10 — Tuning the extended task, and four label-generation conditions at the selected rate

17 September 2026. Local operational report; not tracked in Git.

## Existing files I used, and what I took from them

| Source | What it does | What I took from it |
| --- | --- | --- |
| `CLAUDE.md` | Working conventions | Minimality rule, folder responsibilities, checkpoint policy, outputs to keep and not keep, agent-output limits |
| `Development/reports/09_...md` | Previous stage | The two-condition structure, the separate label-key design, and the bit-for-bit reuse check that justified trusting the refactor |
| `scripts/base_task/tune_learning_rate.py` | The base task's learning-rate search | The whole search shape: grid, reuse-if-already-trained, per-candidate resume, the selection rule and its tie-breaks, the figure. The extended search is the same method applied to the extended model |
| `scripts/extended_task/run_extended.py` | The extended chain | `train_generation`, `build_generation_0_dataset`, `score_dev`, `write_config`, `load_final_model` — all reused by the new search rather than reimplemented |
| `scripts/extended_task/extended_model.py` | The extended model | `generate_continuation`, reused unchanged for both data generation and the new self-generated evaluation |
| `scripts/common.py` | Shared helpers | Paths, `baseline_options()`, `load_features()`, `training_seeds()`, `available_checkpoints()`, `use_above_normal_priority()` |
| `results/extended_task/generation_0/` (as it then was) | The 1e-05 extended generation 0 | Scored as the 1e-05/seed-5 grid cell instead of retraining it, and its training data read as the shared original-task dataset |
| `upstream/` | The authors' code | Untouched. No upstream file was read for modification and none was changed |

## Files created, changed and moved

**Created**

| File | Concrete purpose |
| --- | --- |
| `scripts/extended_task/tune_extended.py` | The extended task's 6 × 3 learning-rate search: trains missing candidates on correct continuations, scores final checkpoints, applies the pre-recorded rule, writes `results.json`, `selection.json` and the figure |
| `findings/04_extended_task_tuning.md` | The search: setup, results, interpretation, limitations |
| `findings/05_label_generation_strategies.md` | The four conditions: the main scientific finding of this stage |
| `temporary_checks/extended_task_at_learning_rate_1e-05/README.md` | What the archive is, why it was superseded, and what did not move with it |

**Changed**

| File | Change | Why |
| --- | --- | --- |
| `scripts/extended_task/run_extended.py` | Results moved under `recursive/`; `selected_learning_rate()` reads the rate from the tuning record; `require_matching_rate()` refuses to reuse a run trained at another rate; `train_generation` takes a checkpoint schedule; `dataset_quality` gained symbol-identity spread and the generated-label counts | The chain had to run at the selected rate, at the agreed paths, without a flag that could drift from the search — and the search needed a lightweight checkpoint schedule from the same trainer |
| `scripts/extended_task/analyse_extended.py` | Added self-generated continuation scoring and explicit ablation *effect* sizes; a fourth per-condition panel with the within-training curves; a six-panel cross-condition figure; reuse of an old `analysis.json` now requires the fields this version reports | The stage asked for teacher-forced and generated measurements kept apart, ablation effects, within-training curves, and a comparison across four rather than two conditions |
| `scripts/extended_task/extended_model.py` | Docstring pointer only | It referenced the finding that moved |
| `scripts/base_task/tune_learning_rate.py` | Docstring pointer only | It cited `findings/03_learning_rate_search.md`, which is `01_` |
| `README.md` | New tuning step, four-condition step, analysis step; findings index; new output paths; checkpoint policy now states that 12 of 55 are analysed; limitations rewritten | It is the public reproduction guide and every path and command moved |
| `CLAUDE.md` | `results/<task>/tuning|recursive` layout; a "learning rates and conditions" section; the archive rule under `temporary_checks/` | These are the standards this stage actually operated under |
| `temporary_checks/README.md` | Now covers two retired items, not one | A second item was archived into it |

**Moved, not copied**

| From | To |
| --- | --- |
| `results/extended_task/{generation_0,experiments,condition_comparison.*}` | `results/extended_task/recursive/…`, then on to the archive |
| the whole 1e-05 recursive experiment | `temporary_checks/extended_task_at_learning_rate_1e-05/results/` |
| `findings/03_extended_task_recursion.md` | `temporary_checks/extended_task_at_learning_rate_1e-05/` |

**Deleted:** two temporary check scripts under `Development/outputs/`, once they had
served their purpose. No timing files, runtime columns, monitoring reports,
duplicate manifests or standalone verification generators were created at any
point. Both requirements files are unchanged.

## Final structure and tracking

```
results/extended_task/
  tuning/
    learning_rate_<rate>_init_seed_<seed>/    18 candidates, 2 checkpoints each
    results.json  selection.json  learning_rate_comparison.png
  recursive/
    generation_0/                             shared by all four conditions
    experiments/
      label_argmax/                           generations 1-4 + chain table and figure
      label_sampling_temperature_1/
      label_sampling_temperature_3/
      label_sampling_temperature_5/
    condition_comparison.json  .png
```

21 MB of tuning, 342 MB of recursive runs, 181 MB archived.

A clone receives `README.md`, `.gitignore`, both requirements files, `scripts/`,
`findings/`, `results/evaluation_data/class_splits.json` and the vendored
`upstream/`. Local and ignored: `Development/`, `temporary_checks/`, `CLAUDE.md`
and the rest of `results/` — all three confirmed ignored with `git check-ignore`.
Nothing was committed, pushed or reset.

**One thing to decide.** `findings/03_extended_task_recursion.md` was already
staged in the Git index from an earlier session. I moved it into
`temporary_checks/`, so `git status` now shows it as deleted in the working tree
while the index still holds the old content. I deliberately did **not** touch the
index. Unstage it before the next commit, or the commit will reintroduce a file
that no longer exists there.

## The tuning search

Six rates (1e-6 … 1e-3) × three initialisation seeds (5, 6, 7) = 18 candidates,
trained on the original extended task with correct continuations, 1,000,000
examples and 31,250 updates each, two checkpoints each.

**What varies with the initialisation seed is the backbone weights only.** The
two-way symbol head keeps its own fixed seed (11) by the existing convention, and
the training data, the development coin flips and the evaluation questions are
identical across all 18. That is recorded in `results.json` rather than left
implicit.

**The rule was on disk before any new result** — though not before every result:
an extended-task model at 1e-05 and seed 5 already existed and its scores were
known, and it became one of the eighteen grid cells. `run_grid` writes `results.json`
— which carries `selection_rule` and `selection_objective` — before training
anything. The rule is highest unrounded mean final **query-label** accuracy over
the three seeds, ties broken by mean query-label loss, then by preferring 1e-05,
then the smaller rate; three completed seeds required. Symbol loss and
following-label accuracy are recorded but do not enter it. Nothing was revised
after seeing which model later collapsed.

**Result: 1e-3**, mean query accuracy 0.9977 against 0.9570 at the authors'
1e-05. All 18 completed; no numerical failures and no implementation failures, so
the distinction never had to be exercised. The winner is at the top of the tested
range, recorded as `winner_at_grid_boundary: true`, and the grid was **not**
expanded in response.

**Reused rather than retrained:** the 1e-05/seed-5 cell. An existing extended
generation 0 had exactly that configuration, so it was scored, not trained again.
Its training data also served as the shared original-task dataset for the other
17 candidates, which avoided both an 18-fold duplication and a rebuild.

## What was archived, and what was kept back

1e-05 did not win, so the whole 1e-05 recursive experiment was moved to
`temporary_checks/extended_task_at_learning_rate_1e-05/` together with its
finding. The finding's internal paths were repointed at the copy of `results/`
beside it, and it carries a header saying it is superseded and that its commands
no longer reproduce it. The folder has a README explaining all of this.

**Before moving anything**, the 1e-05/seed-5 candidate that the *active* tuning
record cites was materialised inside the tuning folder — its `config.json`,
`log.h5` and its initialisation and final checkpoints — and `results.json` was
repointed at that copy. So no active result depends on the archive. That was the
only cross-dependency; it was found by asking what still pointed into the folder
before moving it, not by building any migration machinery.

The base-task experiment was not touched.

## The recursive runs

Generation 0 retrained once at 1e-3, seed 5, with the 55-checkpoint schedule
written directly during training. Its final development scores match the
corresponding tuning candidate to four decimal places on all four measures
(0.9970 / 0.0114 / 0.6934 / 0.9960) — same settings, same deterministic data,
only the checkpoint count differs. **No material discrepancy.**

Then 16 successors: four conditions × four generations, run sequentially in one
background job, argmax first, then temperatures 1, 3 and 5. All four reused the
one generation 0. Total for the stage: 17 tuning trainings plus 17 recursive
trainings, about seven hours of CPU, all at AboveNormal priority.

Generation procedure unchanged from the previous stage except that the label rule
is now a setting: query label by strategy, next symbol always sampled at
temperature 1, following label by strategy, each step conditioned on what the
parent actually produced. Separate key chains for opening questions
(`train_seed`), symbol sampling (14) and label sampling (15), all recorded in
every `config.json` alongside the strategy, the temperature and a plain-English
`label_rule`.

## Results

Wrong query labels per 1,000,000 generated training examples:

| data for gen | argmax | T=1 | T=3 | T=5 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 48 | 50,802 | 220,940 |
| 2 | 0 | 108 | 477,839 | 699,572 |
| 3 | 0 | 140 | 713,796 | 792,942 |
| 4 | 13 | 401 | 788,744 | 799,069 |

Final teacher-forced query-label accuracy, same fixed 1,000 questions:

| gen | argmax | T=1 | T=3 | T=5 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.997 | 0.997 | 0.997 | 0.997 |
| 1 | 0.997 | 0.994 | 0.988 | 0.994 |
| 2 | 0.998 | 0.993 | 0.986 | 0.505 |
| 3 | 1.000 | 0.991 | 0.511 | 0.396 |
| 4 | 0.993 | 0.999 | 0.489 | 0.194 |

Argmax and temperature 1 **remained broadly stable over these four generations** —
no trend in either, and both touched their highest value at generation 3 or 4.
Temperature 3 and temperature 5 collapse, the hotter one a generation earlier.
These are single chains at one seed and budget; the dataset error counts above
describe the data a successor was given, which is a separate measurement from how
that successor then scored.

## Interpretation, and the one thing I did not expect

The interpretation is written up properly in finding 05. The part worth flagging
operationally is that the within-training curves changed what the result means.

Reading only the final numbers, this looks like a circuit being eroded across
generations. Two things rule that framing out. First, **every successor starts
from fresh weights**, so no model inherits a trained circuit; what passes between
generations is the training data. The question is whether each successor
*develops* the induction attention pattern during its own training. Second, the
curves show the stable runs reaching high accuracy early and staying there —
generation 0 first crosses 90% at 30,016 sequences and is above 90% at 97% of its
201 logged evaluation points; the argmax and temperature-1 successors cross
between 50,016 and 65,024, inside the first 7% of their million examples.

First recorded crossing of 90%, from the logged curves:

| condition | gen 1 | gen 2 | gen 3 | gen 4 |
| --- | ---: | ---: | ---: | ---: |
| argmax | 60,000 | 55,008 | 55,008 | 50,016 |
| T=1 | 65,024 | 60,000 | 55,008 | 60,000 |
| T=3 | 150,016 | 560,000 | not reached | not reached |
| T=5 | 225,024 | not reached | not reached | not reached |

In the runs marked "not reached", accuracy never exceeded 0.543 at any of the 201
logged points, and the induction score stayed below 0.05 at all 12 analysed
checkpoints. We did not measure the 43 checkpoints in between, so the honest
statement is that the pattern was absent whenever we looked, not that it never
appeared. And an attention score says where heads attend, not what the circuit
contributes to the output.

The collapsed runs also differ from one another, which the first write-up missed:
temperature 3 generation 3 hovers between 0.21 and 0.53 and ends at 0.511, while
temperature 5 generations 3 and 4 end at 0.396 and 0.194, with generation 4
ranging down to 0.070 — below five-way chance. A successor whose targets were
79.9% wrong was trained towards a nearly uniform label distribution, so scoring at
or below five-way chance against the *true* labels is what that would look like.
That is a proposed explanation, not something this experiment tested.

Corrupted targets first **delay** the crossing — 150,016 then 560,000 at
temperature 3 — before it stops occurring within budget. Whether a longer run
would cross was not tested and these data do not indicate either way, so "within
31,250 updates" is part of the claim rather than a footnote.

I had the within-training panel in the plan only because the stage asked for it.
It turned out to carry the main result, which is an argument for the requirement
rather than for my prioritisation.

Two other things worth recording:

- **The two ablations behave very differently, and neither settles redundancy.**
  Effects are intact-minus-ablated accuracy, so positive means silencing the head
  *hurt*. Silencing the top induction head changed accuracy by between −0.2 and
  +1.2 percentage points anywhere in the experiment (0.3 points in generation 0).
  Silencing the top previous-token head reached **+20.0** and **+29.1 percentage
  points** at temperature 1 generations 2 and 4 (0.993 → 0.793 and 0.999 →
  0.708), both times L0H7. So single-head ablation is not uniformly uninformative
  here; it is small for one selected head and sometimes large for the other. A
  small effect is equally consistent with compensation, with no dependency, and
  with zero-ablation being too weak an intervention — we never silenced heads in
  combination. In the collapsed models every effect is within ±0.3 points, which
  is uninformative because intact accuracy is already near chance.
- **The symbol-identity diversity measure worked; it simply barely moved.** I
  checked the implementation against the saved datasets: it counts the class of
  the symbol the parent actually selected, and normalises by the number of
  examples, so it does measure generated next-symbol identities rather than
  opening contexts or bare class coverage. My first write-up reported the entropy
  rounded to three decimals, which made every dataset look identical and led me
  to claim the design forced the result. At full precision the values do differ:
  largest class share 0.02026 in generation 0's correct data against 0.02050 at
  temperature 3 generation 4, and entropy 3.912006 against 3.911970 nats, with
  ln 50 = 3.912023 as the maximum. The frequencies stayed very close to even, but
  nothing in the design forces that — a parent whose symbol head favoured
  particular identities could have skewed them. No code change was needed and no
  stored measurement changed.

## Remaining issues

1. **Both searches stop at the edge.** 1e-3 is the top of both tested ranges and
   neither grid was extended. Nothing rules out a better or unstable rate above.
2. **One chain per condition from a shared parent.** Not four independent
   replications; a single initialisation seed throughout.
3. **The budget is load-bearing.** "Never formed" means within 31,250 updates.
   Whether a longer run would transition is untested.
4. **Head identity moves within the sampling chains**, so those ablation series
   compare different heads across generations.
5. **The large previous-token ablation effects in the temperature-1 chain**
   (+0.20 at generation 2, +0.29 at generation 4) are unexplained. They are
   recorded, not interpreted.
6. **Why successors transition later than generation 0** in the stable conditions
   (50,000–65,000 against 30,016) is not investigated.
7. The reserved final-test classes remain ungenerated and unscored.

## Corrections pass (17 September 2026)

A review of the whole written record for scientific accuracy. **No experiment was
run, no model retrained, no stored measurement changed.** One temporary script
under `Development/outputs/` resolved the factual questions and was deleted.

**Reviewed:** `findings/01`, `02`, `04`, `05`; `README.md`;
`Development/reports/01`–`10` and the stage-05 handover; `temporary_checks/README.md`,
the archived `03_extended_task_recursion.md` and the archive README; `CLAUDE.md`.

**Changed, and what was wrong:**

| File | Correction |
| --- | --- |
| `findings/05` | Retitled; "the circuit fails to form rather than eroding" narrowed to what the measurements support. Successors start from fresh weights, so nothing is inherited and "erosion across generations" was never the right frame; the question is whether each successor *develops* the pattern. Attention claims now scoped to the 12 analysed checkpoints, accuracy claims to the 201 logged points, with the two densities stated explicitly |
| `findings/05` | The within-training description was wrong in two ways. "Every model sits near 50% for most of training" — the stable runs cross 90% inside the first 7% of their examples and are above 90% at ~94–97% of logged points. "Accuracy ~0.50 from first checkpoint to last" — true of T=3 generation 3, but T=5 generations 3 and 4 finish at 0.396 and 0.194, ranging down to 0.070. Those runs are now described separately, with the reading that a model trained on 79.9%-wrong targets scoring at or below five-way chance is consistent with having learned the corrupted distribution — labelled as a proposed explanation |
| `findings/05` | "A first recorded crossing of 90%" now stated as an operational marker on a logged curve, not the moment a circuit forms; "never formed" replaced by "not reached", with an explicit statement that these data do not indicate whether longer training would help |
| `findings/05` | Symbol-identity section rewritten: the claim that the design makes the measure unable to move was wrong. Values reported at full precision instead |
| `findings/05` | Ablation section rewritten: "single-head ablation measures almost nothing" was false. Sign and units now stated (intact minus ablated, percentage points, positive = silencing hurt), the +20.0 and +29.1 previous-token effects given with their before/after accuracies, and the redundancy inference removed |
| `findings/05` | Dataset error counts now explicitly labelled a property of the data a successor was given, separate from that successor's own scores |
| `findings/02` | "Nothing changed across five generations" scoped to the quantities actually compared |
| `findings/04` | The ln 2 floor now stated as a floor in expectation, since a finite 1,000-question sample can land just below it |
| `README.md` | Ablation limitation rewritten (no redundancy inference, no combination ablations); added limitations for sparse attention sampling and for successors starting from fresh weights; finding-05 index row retitled |
| `Development/reports/10` (this file) | Interpretation section rewritten to match the above |
| `Development/reports/09`, `08` | Superseded headers added pointing at the archive and at finding 05; their own measurements left unchanged. Dataset error rates separated from successor performance; "five identical models" narrowed to "equal wherever compared" |
| `Development/reports/07` | "Reproduced the tuning candidate exactly" narrowed to agreement on the two reported measures |
| `Development/reports/02`, `01`, `04` | The "78,400 possible questions" inference corrected: that is a count of distinct questions appearing in one generated dataset, not the size of the question space, which was never established. Each correction is marked in place and the surrounding conclusions still hold. Also "proves"/"provably" softened to what was actually checked |
| `temporary_checks/.../03_extended_task_recursion.md` | Dataset error rates separated from successor performance; "clearest degradation in the project" marked as true when written; retained class coverage distinguished from unchanged frequencies; ablation sign and units stated |

**Did any measurement change?** No. The symbol-identity entropy was the one
candidate for a real bug, so the implementation was checked against the saved
datasets: `dataset_quality` counts the class of the symbol the parent actually
selected and divides by the number of examples, so it does measure generated
next-symbol identities, not opening contexts and not bare class coverage. The
stored values were correct and full precision; only the three-decimal rounding in
my prose made them look identical, and that led to the mistaken claim that the
design forced the result. At full precision the largest class share runs 0.02026
(generation 0's correct data) to 0.02050 (T=3 generation 4) and entropy 3.912006
to 3.911970 nats against ln 50 = 3.912023. **No code change and no recomputation
were needed.**

**Unresolved, and left as such:**

1. Why silencing L0H7 costs 20.0 and 29.1 percentage points in temperature-1
   generations 2 and 4, but far less elsewhere.
2. Why successors in the stable conditions cross 90% later than generation 0
   (50,016–65,024 against 30,016).
3. Whether the induction pattern appeared at any of the 43 checkpoints we did not
   analyse in the collapsed runs.
4. Whether a longer budget would change any collapsed run.
5. Whether the small differences in symbol-identity frequencies are
   distinguishable from sampling variation.

## Distribution analysis of the generated datasets (17 September 2026)

What changed in the data passed between generations, beyond accuracy and total
error counts. **No training, no generations, no tuning, no model inference** —
every number is counted from the saved `training_data.h5` files.

### Files used and what they contain

All 17 datasets were present (generation 0 plus four generations in each of the
four conditions). Fields, confirmed against the data rather than assumed:

| Field | Shape | Meaning |
| --- | --- | --- |
| `class_idxs` | [n, 3] | opening symbol classes: context A, context B, query |
| `exemplar_idxs` | [n, 3] | which exemplar of each class |
| `labels` | [n, 3] | context A's label, context B's label, and the **generated** query label |
| `symbol_choice` | [n] | 0 or 1 — which context symbol the parent generated next |
| `next_label` | [n] | the **generated** following label |
| `true_query_label` | [n] | correct query label under the original context mapping |
| `true_next_label` | [n] | correct label of the symbol actually chosen |

A temporary check in `Development/outputs/`, since deleted, confirmed four things
the analysis depends on: the query class is always one of the two context classes
(1,000,000 of 1,000,000); the two context classes and the two context labels are
always different, so correct / other-context / outside-context are exclusive and
exhaustive; `true_query_label` and `true_next_label` agree with the context
mapping recomputed from `class_idxs` and `labels` on every row; and **the opening
contexts are byte-identical across all seventeen datasets**, so every class is
offered about 40,000 times as a candidate next symbol everywhere and differences
in what was chosen come from the parent rather than from what it was offered.

### Files changed and why

| File | Change |
| --- | --- |
| `scripts/extended_task/analyse_extended.py` | Added `label_breakdown`, `symbol_breakdown` and `dataset_distributions`, which read a generation's saved dataset and count what it contains; `summarise_condition` now writes those records, and `plot_distributions` draws the cross-condition figure. No new command or flag — the five existing analysis commands produce it |
| `findings/05_label_generation_strategies.md` | New distribution section, the figure, and an interpretation paragraph tying each dataset to the child trained on it |
| `README.md` | Two new output rows and one sentence on what the per-condition analysis now also counts |

New outputs: `dataset_distributions.json` in each condition folder and in
`generation_0/`, plus `data_distributions.png` at the `recursive/` root. Nothing
was duplicated: generation 0 is shared, so its record is written once beside its
own dataset rather than copied into each condition.

### Results

**Labels.** The generated query labels converge on a uniform draw. At temperature
5 generation 4 the correct / other-context / outside-context split is
0.2009 / 0.2016 / 0.5975, which is what choosing one of five labels at random
gives, and the generated marginal is 201,585 / 199,069 / 198,829 / 199,779 /
200,738 against true counts of 188,451 / 187,307 / 187,256 / 187,105 / 249,881.
So the parent did not develop a preference for a label — it **lost** the true
distribution's over-representation of label 4. The correct-versus-generated table
is flat in every row. Temperature 3 generation 4 is close behind but keeps a weak
diagonal (40,590 correct against about 37,000 per wrong label in row 0).

Of the answers that were wrong, about three quarters were not even a context
label, from the *first* corrupted dataset onwards (0.74 at temperature 3
generation 1, 0.73–0.75 thereafter). Uniform choice among the four wrong labels
predicts 0.75. These are not confusions between the two context labels; they are
draws from a flattened distribution, which is what dividing the logits by 3 or 5
does.

**Symbols.** Because every class is offered equally often, the informative
measure is chosen-divided-by-offered per class, which the intended coin flip puts
at 0.5 everywhere. The standard deviation of that rate across the 50 classes is
0.00217 in generation 0's correct data, against a coin-flip sampling spread of
about 0.0025, and runs 0.00231–0.00461 across the sixteen generated datasets —
at or near the floor almost everywhere. The one excursion is temperature 3, at
0.00345 (generation 2) and 0.00461 (generation 4). Temperature 5 corrupted labels
*more* and shows no excursion (0.00231–0.00269), so nothing here indicates that
symbol bias contributed to the label collapse.

**The clear negative result:** generated symbol frequencies stayed close to the
original distribution while label correctness deteriorated completely.

### Was the existing entropy measurement correct?

Yes. `dataset_quality` counts the class of the symbol the parent actually
selected and divides by the number of examples, so it measures generated
next-symbol identities — not opening contexts and not bare class coverage. Values
were stored at full precision and were correct. Rounding did conceal the
differences, but only in the prose: at full precision the entropy runs 3.912006
down to 3.911970 nats against a maximum of ln 50 = 3.912023, so the differences
are real but in the fifth decimal place. No stored measurement was changed. The
new per-class selection rate was added because entropy over a near-uniform
distribution is insensitive, and because it accounts for availability, which
entropy does not.

### Limitations

1. **These are empirical frequencies of generated tokens, not predicted
   probability vectors.** The datasets record what was drawn, not the
   distribution it was drawn from. "The parent's label distribution became
   uniform" is a statement about its outputs on these million questions.
   Recovering the underlying probabilities would need fresh inference over the
   saved checkpoints; that was not needed for any question asked here and was not
   done.
2. **No significance testing.** The temperature-3 symbol excursion is compared
   against a coin-flip sampling spread as a reference, not tested.
3. **The offer counts rely on the two context classes differing**, which holds in
   every row of every dataset here but is a property of these settings.
4. **Association only.** Each dataset is tied to the child trained on it, but
   nothing here tests that a distributional change caused a performance change.
