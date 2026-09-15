# 03 — Full single-generation baseline training and initial analysis

Date: 15 September 2026. Scope: one complete run of the authors' unmodified
induction-head baseline, evaluated with our assignment development protocol,
plus a first mechanistic analysis. No recursive training, extended task,
multi-seed sweep or intervention suite.

## Outcome

The full run **completed**: 1,000,000 training sequences, 31,250 optimizer
updates, exit code 0, in **5.59 minutes** of wall time on CPU. Development
accuracy on 100 unseen character classes rose from 20.9% to **96.7%**, so the
model genuinely learned in-context learning that transfers to symbols it never
saw in training.

Mechanistically, the run is **qualitatively consistent with** the behaviour the
paper describes. We did not compare against any published number, so this is not
a reproduction of their results. Previous-token heads in layer 0 and induction
heads in layer 1 both emerge in a sharp transition between roughly 150,000 and
400,000 sequences, and the accuracy jump happens in that same window. The
contribution appears **distributed**: seven of eight layer-1 heads end with a
clearly positive induction score, and ablating the strongest one costs only a
few accuracy points. The ablations also show that attention-pattern strength and
causal contribution are not the same thing.

The reserved final test was not built and not scored.

> **Note added in stage 04.** Files and run folders were reorganised after this
> report was written. Commands and paths below are left exactly as they were run,
> as historical provenance. Current locations:
> `scripts/run_baseline.py` -> `scripts/base_task/train_original.py`;
> `scripts/analyse_baseline.py` -> `scripts/base_task/analyse_runs.py`;
> `scripts/make_assignment_evaluators.py` -> `scripts/prepare_evaluation_data.py`;
> `results/baseline_full_is5_assignment/` ->
> `results/base_task/generation_0_original_data_init_seed_5/`;
> `results/assignment/` -> `results/evaluation_data/`.
> Stage 04 also corrected wording in this report; see
> `reports/04_project_cleanup_and_first_successor.md`. No number changed.

## Code used

### Upstream (`upstream/icl-dynamics/`, unmodified)

| File / function | Role in this run |
| --- | --- |
| `main.py` `run_with_opts` (:215) | The whole training run: splits, samplers, evaluators, model, optimizer, loop, checkpointing |
| `main.py` `train_step` (:87), `compute_loss` (:70), `ce` (:59) | One Adam update; loss is cross-entropy at the query position only |
| `main.py` `evaluate` (:200), `eval_step` (:132) | Scores each fixed evaluator; source of `acc`, `in_context_acc`, `prob`, `use_context_prob` |
| `main.py` evaluator loading (:357-369) | Injects our `fsl_dev_class` from file — how the assignment protocol works without patching |
| `main_utils.py` `get_splits_from_opts` (:153), `get_model_from_opts` (:222), `get_optimizer_from_opts` (:250) | Splits, the 2-layer/8-head attention-only model, constant-LR Adam |
| `samplers.py` `get_constant_burst_seq_idxs` (:32), `get_exemplar_inds` (:167), `fewshot_relabel` (:221), `make_data_sampler` (:288) | Sequence construction and per-sequence label reassignment |
| `models.py` `SequenceClassifier` (:596, interleaving at :748-753), `AttentionBlock` (:260) | 5-token interleaving with the query label dropped; caches `attn_scores`, which every measure below reads |
| `visualize_runs.py` `make_forward_fn` (:701) | Forward pass returning attention `[batch, layer, head, query, key]` plus loss/accuracy |
| `visualize_runs.py` `update_prev_token_over_time` (:338), `plot_prev_token_over_time` (:355-364), `plot_attention_over_time` (:513-526) | The authors' definitions of the two progress measures |
| `opto.py` `make_fn_from_opts` (:150), ablation (:368-372) | Head ablation by zeroing value vectors |

### Ours

| File | Role | Created/modified |
| --- | --- | --- |
| `scripts/run_baseline.py` | Launched the run; `--full --protocol assignment` | Modified in stage 02 |
| `scripts/make_assignment_evaluators.py` | Produced the dev evaluator and split record used here | Created in stage 02 |
| `scripts/analyse_baseline.py` | All curves, progress measures, attention figure and ablations in this report | **Created in this stage** |

`analyse_baseline.py` deliberately calls the authors' `make_forward_fn` rather
than reimplementing the forward pass. Its two measure functions were checked
against the authors' own update functions on the same checkpoint and agree to
within 1e-6 (see Verification).

## Configuration and command

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py --full --protocol assignment --run-name baseline_full_is5_assignment
.\.venv\Scripts\python.exe scripts/analyse_baseline.py results/baseline_full_is5_assignment
```

The underlying arguments are the first `python main.py` line of
`ih_paper_runs.sh` (Figure 3a, `omniglot50_rl5`) with `MAIN_RUN_ITERS=1000000`
and `INIT_SEED=5`; the exact vector is saved in
`results/baseline_full_is5_assignment/command.json`.

### Original settings, kept

2 attention-only transformer layers, 8 heads each, residual dimension 64, no
MLP blocks, RoPE positions, 66,629 parameters; 50 training classes, exemplar 0,
two symbol–label pairs plus a supported query, 5 output labels with per-sequence
relabeling from the 8 training label pairs; batch size 32; constant Adam at
1e-5; initialization seed 5, training seed 0, evaluation seed 1, label-pair
split seed 20; 1,000,000 sequences = 31,250 updates; evaluation every 5,000
sequences; checkpoint every 1,000 sequences. Fixed duration — no early stopping
and no hyperparameter tuning.

### Our evaluation adaptation

| | Authors | This run |
| --- | --- | --- |
| Evaluators | `fsl_train`, `fsl_val_rl`, `fsl_train_valex`, `fsl_test_class` | `fsl_train`, `fsl_val_rl`, `fsl_train_valex`, **`fsl_dev_class`** |
| Held-out classes scored | rows 1523–1622, scored throughout training | 100 rows drawn from the unused 50–1522 pool |
| Final test | none reserved | 100 further disjoint rows **reserved, never scored** |

Training was unaffected by the evaluation change. Stage 02 checked this with one
pair of 3,200-sequence runs, one per protocol, which produced checkpoints with
identical hashes. That is direct evidence for those runs, plus a mechanism that
supports the general case, rather than a proof covering every configuration.

## Completion and runtime

| Item | Value |
| --- | --- |
| Return code | 0 |
| Sequences processed | 1,000,000 (final checkpoint `iter` = 1000000) |
| Optimizer updates | 31,250 (restored Adam `count` = 31,250) |
| `train_loss` entries logged | 31,250 |
| Wall time incl. imports, compilation, evaluation, checkpoint writes | **335.14 s = 5.59 min** |
| Upstream internal timer | 5.55 min |
| First evaluation marker | 8.14 s after launch |
| Checkpoints written | 1,001 (0 … 1,000,000), 809,283 bytes each |
| Run folder size | 795 MB (checkpoints 775 MB, `log.h5` 20 MB) |
| Disk free afterwards | 67 GB |

The original checkpoint schedule was kept unchanged; the 810 MB estimate in
report 01 was accurate (795 MB actual). The run was much faster than report 01's
cautious estimate of "minutes to tens of minutes" — the extrapolation from a
100-update sample overstated per-update cost. The run was not interrupted, so no
resume procedure was needed.

## Results

### Learning curves

Each figure is `results/baseline_full_is5_assignment/analysis/curves.png`;
numbers are means over each evaluator's 1,000 fixed sequences.

| Evaluator | What it measures | Initial acc | Final acc | Initial loss | Final loss |
| --- | --- | ---: | ---: | ---: | ---: |
| `fsl_dev_class` | **Our development validation: 100 unseen classes** | 20.9% | **96.7%** | 2.0038 | 0.0828 |
| `fsl_train` | Training-distribution diagnostic (*not* guaranteed-disjoint validation) | 20.7% | 100.0% | 1.9366 | 0.0015 |
| `fsl_val_rl` | Unseen label *pairings*, training classes | 22.9% | 99.6% | 2.0433 | 0.0265 |
| `fsl_train_valex` | Unseen *exemplars* of training classes | 21.6% | 86.2% | 1.9813 | 0.5933 |

Training loss fell from 1.9062 (first 10 batches) to 0.0011 (last 10); minimum
0.00035. All gradient norms were finite, ranging 0.019 to 18.644.

Development accuracy first exceeded 60% at 190,016 sequences, 80% at 265,024,
90% at 355,008 and 95% at 580,000.

**Chance levels.** `acc` is argmax over all five labels, so uniform guessing is
**20%**. `in_context_acc` restricts the argmax to the two labels actually
present in the context, so its chance level is **50%**. Out-of-context accuracy
is necessarily zero here because the correct label always appears in context.

A useful detail: by about 25,000 sequences `acc` and `in_context_acc` had both
converged to ~50%. That means the model had already learned to put its mass on
the two labels present in context — but could not tell *which* of the two was
correct. It sat at that ~50% plateau until the induction circuit formed. At the
end the two metrics coincide again (96.7%), because almost all probability mass
now sits on in-context labels.

**Observation.** `fsl_train_valex` is the one evaluator whose loss degrades: it
bottoms out near 0.47 around 400,000 sequences and then **rises** to 0.593,
while its accuracy stays flat at ~86%.

**Proposed explanation, not tested here.** Rising loss at flat accuracy is the
signature of a model growing more confident on the examples it already gets
wrong, which would be ordinary overfitting to the exemplar-0 training
distribution. We did not measure the per-example probabilities that would
confirm it, so this remains a hypothesis. It is **not** model collapse: this is
a single ordinary training run on real data, with no generated data in it.

### Mechanistic measurements

Figure: `analysis/head_measures.png`. Fifteen checkpoints spanning the run were
analysed, spaced roughly logarithmically (0; 1,024; 2,016; 3,008; 5,024; 9,024;
14,016; 25,024; 42,016; 71,008; 120,000; 204,000; 347,008; 589,024; 1,000,000).

**Previous-token score (layer 0).** Raw value is the attention *a* that a token
gives to its immediate predecessor. Following the authors, we subtract a chance
baseline. Write **r** for the index into the shortened array the authors build
with `inds = arange(1, seq)`, so r = 0 is original token position 1, r = 1 is
position 2, and so on. The corrected score is **a − (1 − a) / (1 + r)**.

The denominator `1 + r` is the number of causally visible positions *other than*
the previous token. Worked through for r = 1: that is token position 2, which can
attend to positions {0, 1, 2}; the previous token is position 1, leaving {0, 2},
two other positions, and 1 + r = 2. If attention to the previous token is
a = 0.60, the remaining 0.40 spread over those two positions gives a chance
share of 0.20, so the corrected score is 0.60 − 0.20 = 0.40. The measure is
exactly zero when attention is uniform over everything a token can see, which we
checked at every position.

Zero therefore means "no more than chance". We report the average over the *label* tokens
(positions 1 and 3), the authors' "Average for 1,3" row — this is the part the
induction circuit needs, since a label token must carry information about the
symbol before it.

**Induction score (layer 1).** At the query token, the correct label token is
at position 2·correct_ind+1 and the incorrect one at 2·(1−correct_ind)+1, where
correct_ind says which context pair contains the support symbol. We report the
authors' delta, attention(correct) − attention(incorrect). Zero means the head
cannot distinguish the matching label from the distractor label; +1 would be
perfect induction. `correct_ind` is derived from the data itself: with exemplar
0 only, a support symbol vector is bit-identical to the query symbol vector.

| Sequences | Dev acc | Dev loss | Max induction delta (L1) | Max prev-token score (L0) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.209 | 2.0038 | +0.005 | +0.220 |
| 9,024 | 0.423 | 1.3792 | +0.007 | +0.175 |
| 25,024 | 0.501 | 0.9161 | +0.011 | +0.076 |
| 71,008 | 0.510 | 0.7668 | +0.015 | +0.034 |
| 120,000 | 0.532 | 0.7432 | +0.017 | +0.052 |
| 204,000 | 0.632 | 0.6756 | +0.081 | +0.295 |
| 347,008 | 0.893 | 0.2744 | +0.429 | +0.784 |
| 589,024 | 0.947 | 0.1323 | +0.597 | +0.866 |
| 1,000,000 | 0.967 | 0.0828 | +0.686 | +0.878 |

Three phases are visible:

1. **Early non-induction learning (0 – ~25k).** Accuracy climbs 21% → 50% while
   both circuit measures stay flat. The model is learning to restrict output to
   in-context labels, not to match symbols. The previous-token score actually
   *decays* from its initialization value (+0.220 → +0.034); the small initial
   value is a property of initialization and RoPE, not something learned.
2. **Plateau (~25k – ~150k).** Accuracy stuck near the 50% two-label chance
   level; induction delta still ≈ +0.017.
3. **Circuit formation (~150k – ~400k).** Both measures rise and accuracy jumps
   53% → 89%. After 400,000 sequences everything continues improving slowly.

   In our sampled checkpoints the previous-token score is already well above its
   plateau at 204,000 sequences (+0.295) while the induction score is still small
   (+0.081), which is *consistent with* previous-token heads leading. It does not
   establish that ordering: consecutive analysed checkpoints are tens of thousands
   of sequences apart, the two measures are on different scales and so cannot be
   ranked against each other by size, and this is one seed. All 1,001 checkpoints
   are on disk, so a denser scan could test the ordering properly.

Final per-head values:

| Head | Induction delta (L1) | | Head | Prev-token score (L0) |
| --- | ---: | --- | --- | ---: |
| L1H3 | **+0.686** | | L0H2 | **+0.878** |
| L1H1 | +0.571 | | L0H5 | +0.759 |
| L1H2 | +0.549 | | L0H1 | +0.705 |
| L1H7 | +0.440 | | L0H4 | +0.052 |
| L1H5 | +0.398 | | L0H7 | −0.034 |
| L1H0 | +0.312 | | L0H6 | −0.347 |
| L1H4 | +0.284 | | L0H3 | −0.370 |
| L1H6 | +0.085 | | L0H0 | −0.449 |

Seven of eight layer-1 heads develop a substantial induction score, and three
layer-0 heads become strong previous-token heads. Candidate heads for ablation
were chosen from these observed values, **not** from the authors' hardcoded
indices.

### Attention example

`analysis/attention_example.png` shows all 16 heads at the final checkpoint on
one development sequence whose support is symbol B (target label 0). The
induction pattern is directly visible: in layer 1 the query row is bright on
"lab B" — the label token immediately following the matching support — for
L1H1, L1H2, L1H3 and L1H7. In layer 0, L0H1, L0H2 and L0H5 show the
sub-diagonal band of previous-token attention. Rows attend to columns and every
row sums to 1; the upper triangle is zero, confirming causal masking.

### Head ablation

Ablation here means the authors' `--opto_ablate_heads`: the named head's **value
vectors are set to zero**, so it writes nothing into the residual stream, while
its attention pattern is still computed and every other head and weight is
untouched. This isolates that head's contribution to the output. All four
conditions were scored on the same 1,000 development sequences at the final
checkpoint.

| Condition | Head | Dev acc | Δ acc | Dev loss |
| --- | --- | ---: | ---: | ---: |
| Intact | — | 96.70% | — | 0.0828 |
| Candidate induction head ablated | L1H3 (highest induction score) | 92.80% | −3.90 | 0.1859 |
| Comparison layer-1 head ablated | L1H6 (lowest induction score) | 96.60% | −0.10 | 0.0816 |
| Candidate previous-token head ablated | L0H2 (highest prev-token score) | 88.90% | −7.80 | 0.2999 |
| Comparison layer-0 head ablated | L0H0 (lowest prev-token score) | 93.10% | −3.60 | 0.2136 |

Three readings:

* **The comparison in layer 1 is clean.** Removing the head with the weakest
  induction pattern changes nothing (−0.1 points, loss marginally lower);
  removing the strongest costs 3.9 points and more than doubles the loss. The
  induction score therefore tracks something causally real.
* **The contribution looks distributed.** Deleting the strongest induction head
  still leaves 92.8% accuracy, and seven heads carry positive induction scores.
  That points to redundancy, and it means calling any one head "the" induction
  head of this model would be unsupported. It does **not** prove that no
  individual head is necessary: we ablated four of the sixteen heads, one at a
  time, in one model. Establishing necessity would need every head tested and
  combinations ablated, since two heads can each be individually dispensable
  while the pair is not.
* **Attention pattern ≠ causal contribution.** L0H0 has the *most negative*
  previous-token score, yet ablating it costs 3.6 accuracy points — nearly as
  much as the best induction head. A head that does not implement the
  previous-token pattern is still doing useful work. This is exactly why pattern
  evidence must not be reported as a causal claim, and it is a caution for the
  project's planned circuit-strength measures.

## Output paths

All under `results/baseline_full_is5_assignment/`:

| Path | Contents |
| --- | --- |
| `config.json`, `command.json` | Resolved options and the exact argument vector |
| `log.h5` | Raw metrics: 31,250 training losses/grad norms, 201 evaluation points per evaluator |
| `checkpoints/00000000000.eqx` … `00001000000.eqx` | 1,001 checkpoints (model, optimizer state, PRNG keys) |
| `console.log`, `timing.json` | Full stdout and 201 timed events |
| `analysis/analysis.json` | Every number quoted above |
| `analysis/curves.png`, `head_measures.png`, `attention_example.png` | The three figures |

The protocol record is `results/assignment/class_splits.json` (tracked);
`results/assignment/eval_dev.h5` holds the fixed development sequences.

## Verification

| Check | Result |
| --- | --- |
| Final checkpoint loads | Yes, `iter` = 1,000,000 |
| Optimizer update count | 31,250 = 1,000,000 / 32, as configured |
| Parameters finite, count | Yes; 66,629, matching report 01 |
| Parameter change from initialization | L2 = 3.7535 |
| Training log length | 31,250 entries, all gradient norms finite |
| Our prev-token measure vs `update_prev_token_over_time` | Identical to 1e-6 |
| Our induction measure vs `plot_attention_over_time` logic | Identical to 1e-6 |
| Attention rows sum to 1; upper triangle zero | Yes (softmax, causal) |
| `fsl_test_class` present in log | **No** — confirmed absent |
| Final test built or scored | **No** — `eval_final_test.h5` not generated |

## Limitations and unresolved issues

* **One seed, one run.** Initialization seed 5 only. The timing of the phase
  transition and which heads specialise are almost certainly seed-dependent.
  This is not multi-seed robustness and not a reproduction of the paper's
  figures — no paper numbers were compared against.
* **Nothing here is model collapse.** This is a single generation trained on
  original data. No recursive training has been run.
* **Ablation is single-head and zero-ablation only.** Four of sixteen heads were
  ablated, one at a time. Redundancy is therefore inferred, not measured: no
  claim about necessity or sufficiency of any head follows from these results.
  Zero-ablation also moves activations off-distribution; mean ablation would be
  a fairer control and was not run.
* **Fifteen checkpoints**, so the transition's onset is located to within tens
  of thousands of sequences, not precisely. All 1,001 checkpoints are on disk if
  a finer scan is wanted.
* **The `fsl_train_valex` loss increase** after ~400,000 sequences is
  unexplained here.
* **L0H0's causal importance without a previous-token pattern** is unexplained
  and would need a different measure to characterise.
* **Generated evidence is not in version control.** The 795 MB run folder is
  ignored by Git, so these results survive only on this machine; the numbers
  quoted in this report are the durable record. Re-running the exact command
  regenerates them.
* The upstream commit cannot be re-verified locally because upstream is
  vendored rather than a submodule (see report 02).

## Next recommended step

The baseline is established and the protocol is in place, so the sensible next
step is the **first recursive generation**: sample a training set from this
model's own predictions, train generation 2 from the same initialization and
settings, and score it with the *same* fixed development evaluator so
generations are directly comparable. Because `analyse_baseline.py` already
reports both progress measures and ablations per run, the project's central
question — whether induction-circuit function weakens before accuracy declines —
can be asked directly by comparing generation 1 and generation 2 on identical
axes.

Two things to decide before running it: how many sequences each generation
generates and whether label reassignment is resampled or inherited; and whether
generation 2 starts from initialization seed 5 or from generation 1's weights.
Keeping seed 5 and fresh initialization makes generations comparable to this
run. The extended task (predicting the next symbol and its label) should come
after a working single-generation recursive loop, not simultaneously.

Attribution: [Singh et al., *What needs to go right for an induction head? A
mechanistic study of in-context learning circuits and their formation*](https://arxiv.org/abs/2404.07129).
Measure definitions come from the authors' code as cited; all numbers come from
this run.
