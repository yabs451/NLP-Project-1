# Finding 01 — The baseline learns in-context learning, and an induction circuit forms with it

Generation 0 | 15 September 2026 | source run:
`results/base_task/generation_0_original_data_init_seed_5`

> **Retired, 16 September 2026.** This run used the authors' learning rate of
> 1e-5. The learning-rate search (finding 03) selected 1e-4, so this model is no
> longer the project's generation 0 and its lineage was retired. Its checkpoints
> and `log.h5` were deleted; the analysis outputs cited below were kept, so the
> numbers and figures here remain valid, but they can no longer be regenerated
> without retraining. The one model still in use from this run is its final
> checkpoint, which serves as the 1e-5 / seed-5 tuning candidate and now lives at
> `results/base_task/tuning/learning_rate_1e-05_init_seed_5/`.

## Question

Does the authors' small transformer, trained unchanged on the symbol–label
in-context-learning task, learn to generalise to **character classes it has
never seen**? And can we see the induction circuit that is supposed to be doing
the work?

This is the parent model for everything that follows, so we also need to know
how good it is before we ask it to generate training data.

## Setup

The authors' baseline, unmodified: 2 attention-only transformer layers, 8 heads
each, 64-dimensional residual stream, 66,629 parameters. Each sequence is two
symbol–label pairs plus a query symbol that matches one of them; the model
predicts the query's label out of 5. Labels are reassigned randomly per
sequence, so the task can only be solved from context. Training uses the
original 50 character classes and exemplar 0, batch size 32, constant Adam at
1e-5, for 1,000,000 sequences (31,250 updates). Initialisation seed 5, training
seed 0.

Evaluation uses our own protocol rather than the authors' diagnostics:
**`fsl_dev_class`**, 1,000 fixed sequences built from 100 character classes
drawn from the 1,473 that no original evaluator touches. A further 100 disjoint
classes are reserved as a final test and have never been scored. The authors'
three training-class diagnostics are kept alongside.

## Results

| Evaluator | What it measures | Start | End |
| --- | --- | ---: | ---: |
| `fsl_dev_class` | **100 unseen classes (our validation)** | 20.9% | **96.7%** |
| `fsl_train` | training distribution (diagnostic only) | 20.7% | 100.0% |
| `fsl_val_rl` | unseen label pairings | 22.9% | 99.6% |
| `fsl_train_valex` | unseen exemplars of training classes | 21.6% | 86.2% |

Chance is 20% over five labels, or 50% once predictions are restricted to the
two labels present in context. Final development loss 0.0828 nats.

![learning curves](../results/base_task/generation_0_original_data_init_seed_5/analysis/curves.png)

Learning has three phases (`results/base_task/generation_0_original_data_init_seed_5/analysis/head_measures.png`):

1. **0 – ~25k sequences.** Accuracy climbs 21% → 50% while both circuit measures
   stay flat. The model learns to put its mass on the two labels present in
   context, without knowing which is right.
2. **~25k – ~150k.** A plateau at the 50% two-label chance level.
3. **~150k – ~400k.** Previous-token and induction scores rise together and
   accuracy jumps 53% → 89%, reaching 96.7% by the end.

Final per-head measures: seven of eight layer-1 heads end with a positive
induction score (strongest L1H3 at +0.686, weakest L1H6 at +0.085), and three
layer-0 heads become strong previous-token heads (L0H2 +0.878, L0H5 +0.759,
L0H1 +0.705).

The induction pattern is directly visible in the attention maps
(`results/base_task/generation_0_original_data_init_seed_5/analysis/attention_example.png`): at the final checkpoint the
query token attends to the label that follows the matching support symbol.

Ablating one head at a time at the final checkpoint (zeroing its value vectors,
so it writes nothing into the residual stream):

| Ablated | Dev accuracy | Change |
| --- | ---: | ---: |
| nothing (intact) | 96.7% | — |
| L1H3, strongest induction score | 92.8% | −3.9 |
| L1H6, weakest induction score | 96.6% | −0.1 |
| L0H2, strongest previous-token score | 88.9% | −7.8 |
| L0H0, most *negative* previous-token score | 93.1% | −3.6 |

## Interpretation

The model genuinely learns the in-context rule rather than memorising symbols:
96.7% on characters it never trained on, with the same accuracy whether or not
predictions are restricted to in-context labels.

The induction score tracks something causally real — removing the head with the
weakest pattern changes nothing, removing the strongest costs real accuracy.

But two cautions come straight out of the same table. First, the contribution is
**distributed**: deleting the strongest induction head still leaves 92.8%, so no
single head can be called "the" induction head here. Second, **an attention
pattern is not a causal contribution**: L0H0 has the most negative
previous-token score of any head yet ablating it costs 3.6 points, nearly as
much as the best induction head. Pattern-based measures must not be reported as
causal claims.

## Limitations and unresolved

- **One seed, one run.** The timing of the transition and which heads specialise
  are likely seed-dependent.
- **Not a reproduction of the paper.** We never compared against a published
  number; the behaviour is only qualitatively consistent with it.
- **Ordering of head formation is suggestive, not established.** The
  previous-token score is ahead of the induction score at 204k sequences, but
  analysed checkpoints are tens of thousands of sequences apart and the two
  measures are on different scales, so they cannot be ranked by size.
- **Redundancy is inferred, not measured.** Four of sixteen heads were ablated,
  one at a time. Nothing here establishes that no head is necessary; that would
  need every head tested and combinations ablated.
- **Unexplained:** `fsl_train_valex` loss bottoms out near 0.47 at ~400k
  sequences and then rises to 0.593 while its accuracy stays flat at ~86%. The
  obvious guess is ordinary overfitting — growing confidence on examples it
  already gets wrong — but we did not measure the per-example probabilities that
  would confirm it. This is a single ordinary run on real data and has nothing
  to do with model collapse.

## Reproduce

From the project root, with the environment set up (see README):

```powershell
.\.venv\Scripts\python.exe scripts/prepare_evaluation_data.py
.\.venv\Scripts\python.exe scripts/base_task/train_original.py --run-name generation_0_original_data_init_seed_5
.\.venv\Scripts\python.exe scripts/base_task/analyse_runs.py results/base_task/generation_0_original_data_init_seed_5
```

This run was trained at the authors' learning rate of 1e-5, before the
learning-rate search in finding 03. It was originally saved with 1,001
checkpoints; it now keeps the 64 that the figures and the checkpoint policy
need, and the analysis above regenerates from them unchanged.
Seeds: initialisation 5, training 0, evaluation 1, label-pair split 20, our
class split 7. Full numbers: `results/base_task/generation_0_original_data_init_seed_5/analysis/analysis.json`.
Operational detail: `Development/reports/03_full_baseline_training_and_analysis.md`
(local only, not tracked).
