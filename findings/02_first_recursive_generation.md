# Finding 02 — One generation of label-only recursion changes essentially nothing

Generation 1 | 15 September 2026 | source run:
`results/base_task/generation_1_generated_data_init_seed_5`
Parent: generation 0 (`..._generation_0_original_data_init_seed_5`)

## Question

If we train a fresh model on the *parent's own answers* instead of the true
answers, does anything degrade — accuracy, or the induction circuit underneath
it? This is the first step of the recursive-training loop the project is built
around, so the first thing to establish is whether the pipeline works and what
one generation actually costs.

## Setup

Generation 1 was produced in three steps.

1. **Questions** come from the *original* training distribution, unchanged: the
   same 50 training character classes, exemplar 0, two symbol–label pairs plus a
   supported query, one distractor, and the same 8 training label pairs. The
   development and reserved final-test classes are never used. We reuse the
   baseline's own training key chain, so generation 1 sees **the same 1,000,000
   questions in the same order** that generation 0 was trained on.
2. **Answers** come from the parent. Generation 0's final checkpoint is run
   without weight updates, and its **argmax over all five output labels** becomes
   the new training target. Nothing is sampled, filtered, or mixed with true
   answers. Only the query's target is replaced — context labels and input
   symbols are untouched, and the target never enters the model's input.
3. **Training** is a fresh model from initialisation seed 5 with a fresh
   optimizer — *not* the parent's weights — for exactly one pass over the saved
   million examples: 31,250 updates at batch size 32. Same architecture,
   optimizer, learning rate and batch size as the parent. No tuning, no early
   stopping.

Evaluation is unchanged from generation 0: the same fixed 1,000-sequence
`fsl_dev_class` evaluator on 100 unseen classes, plus the three training-class
diagnostics. The reserved final test was not built and not scored.

## Results

### The parent's answers were almost all correct

| Measure | Value |
| --- | ---: |
| Generated training examples | 1,000,000 |
| Parent argmax **wrong** vs the true answer | **0.0013%** (13 of 1,000,000) |
| Parent predicted a label **absent from context** | **0.0000%** (0) |
| Distinct questions covered | 78,400 (every possible one, ~12.8× each) |

The 13 errors are not scattered noise. They are all **the same question**, which
appears 13 times and is mislabelled every time — the parent is deterministic, so
a question it gets wrong it gets wrong consistently. Exactly **one of 78,400
distinct questions** is systematically corrupted.

**A correct example** (dataset index 0): context is class 13 → label 2 and class
44 → label 4; the query is class 13; the true answer is 2. The parent puts
0.9998 on label 2 and that becomes the training target.

**The mislabelled example** (dataset index 5257): context is class 11 → label 0
and class 18 → label 4; the query is class 11, so the true answer is **0**. The
parent puts 0.221 on label 0 but **0.778 on label 4** — the distractor's label —
so **4** is stored as the training target. The parent is confidently wrong, and
it copies the wrong context item's label.

### Generation 1 is very close to generation 0, but not identical to it

| Measure | Generation 0 | Generation 1 |
| --- | ---: | ---: |
| Dev accuracy (100 unseen classes, 1,000 questions) | 96.70% | **96.70%** |
| Dev context-restricted accuracy | 96.70% | 96.70% |
| Dev loss (vs **true** answers) | 0.0828 | 0.0833 |
| `fsl_train` accuracy | 100.0% | 100.0% |
| `fsl_val_rl` accuracy | 99.6% | 99.6% |
| `fsl_train_valex` accuracy | 86.2% | 86.2% |

Note the two losses are against different things: generation 1's **training**
loss is measured against the parent's generated targets, while its
**development** loss is against true answers, exactly as for generation 0.

**The two models are close, not identical.** Both answered exactly 967 of the
same 1,000 development questions correctly, but development loss differs by
0.000493 nats, the per-head induction scores by up to 0.0015, two of the five
ablation conditions by 0.1 accuracy point, and the final weights by a measurable
amount (parameter change from initialisation 3.7535 against 3.7543). Those
differences are real, not rounding. What can be said is that they are far too
small for this evaluator to resolve: at 1,000 questions the standard error on an
accuracy near 97% is about 0.54 points, so equal accuracy here means "closer
than we can measure", not "the same model". Stage 05 added a 10,000-question
development set for exactly this reason.

![generation 1 learning curves](../results/base_task/generation_1_generated_data_init_seed_5/analysis/curves.png)

The learning curves are visually superimposable, including the ~50% plateau and
the transition at 150k–400k sequences.

### The induction circuit is unchanged

Candidate heads were selected from **each model's own final scores**, not
inherited from the parent. Independently, both models selected the same heads —
L1H3 (strongest induction), L1H6 (weakest), L0H2 (strongest previous-token),
L0H0 (most negative previous-token). That agreement is itself a result, and it
is what makes the comparison below meaningful.

| Head | Induction score gen 0 | gen 1 | Head | Prev-token gen 0 | gen 1 |
| --- | ---: | ---: | --- | ---: | ---: |
| L1H3 | +0.6859 | +0.6856 | L0H2 | +0.8784 | +0.8786 |
| L1H1 | +0.5711 | +0.5710 | L0H5 | +0.7593 | +0.7592 |
| L1H2 | +0.5486 | +0.5483 | L0H1 | +0.7051 | +0.7049 |
| L1H6 | +0.0846 | +0.0840 | L0H0 | −0.4487 | −0.4479 |

Largest difference across all sixteen heads: 0.0015 for induction, 0.0010 for
previous-token. Small, but measurable and consistent in sign.

Ablations at the final checkpoint (each head's value vectors zeroed):

| Ablated | Gen 0 dev acc | Gen 1 dev acc |
| --- | ---: | ---: |
| nothing | 96.7% | 96.7% |
| L1H3 (strongest induction) | 92.8% | 92.8% |
| L1H6 (weakest induction) | 96.6% | 96.7% |
| L0H2 (strongest previous-token) | 88.9% | 88.8% |
| L0H0 (most negative previous-token) | 93.1% | 93.1% |

## Interpretation

**One generation of label-only recursion produced no measurable degradation,
and that is the expected result rather than a surprise.** The parent scores 100%
on the training distribution, so its answers reproduce the true task for
99.9987% of examples. Generation 1 was therefore trained on almost exactly the
task generation 0 was trained on, and it learned almost exactly the same
solution: the same accuracy to within this evaluator's resolution, visually
identical curves, the same heads selected, and an ablation profile differing by
at most 0.1 accuracy point.

The measurable effect of recursion here is one corrupted question out of 78,400.
That single question is the entire difference between the two training sets.

Two things this does establish:

- **The pipeline works** end to end, and the comparison is tightly controlled:
  identical questions, identical order, identical initialisation, so any
  difference between generations is attributable to the changed targets alone.
- **The parent's error is systematic, not random.** It confidently copies the
  wrong context label for one specific class pair. Errors of that kind are the
  ones recursion could plausibly amplify over many generations, because every
  future generation is trained on the same wrong answer every time that question
  appears.

**What it does not establish.** Two generations cannot say anything about
whether circuit function degrades before accuracy does. That claim needs a
sequence of generations long enough for something to actually change. Nothing
here is model collapse, and we did not adjust any setting to try to produce it.

## Limitations and unresolved

- **Stability may be an artefact of a near-perfect parent.** With a 0.0013%
  error rate there is almost nothing to propagate. Degradation may need many
  generations, a weaker parent (fewer training sequences), or the extended task,
  where generated *symbols* introduce real distributional drift.
- **Generation 1 is not an independent sample.** Reusing the parent's question
  stream and model seed maximises control but means the two runs are not
  statistically independent; the agreement is partly by construction. A fresh
  question stream would trade control for independence.
- **One seed.** Both generations use initialisation seed 5.
- **Unresolved:** why the parent fails specifically on classes 11 vs 18 with
  context labels {0, 4}. We did not investigate whether those two feature
  vectors are unusually similar, which is the obvious first check.
- The `fsl_train_valex` loss rise noted in finding 01 reappears identically in
  generation 1 (0.5933 → 0.5943) and is still unexplained.

## Reproduce

From the project root:

```powershell
.\.venv\Scripts\python.exe scripts/base_task/run_recursive.py --parent results/base_task/generation_0_original_data_init_seed_5 --generations 1
.\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py results/base_task/generation_1_generated_data_init_seed_5
```

The generated dataset is 4.3 MB. This successor was trained from a parent
trained at the authors' learning rate of 1e-5, before the learning-rate search
in finding 03, so it is a pilot rather than part of the tuned lineage. It was
originally saved with 1,001 checkpoints and now keeps the 64 the figures and
the checkpoint policy need; the analysis above regenerates from them unchanged. Seeds: initialisation 5, training 0, evaluation 1.
Full numbers: `results/base_task/generation_1_generated_data_init_seed_5/analysis/analysis.json` and
`results/base_task/generation_1_generated_data_init_seed_5/generation_metadata.json`. Operational detail:
`Development/reports/04_project_cleanup_and_first_successor.md` (local only,
not tracked).
