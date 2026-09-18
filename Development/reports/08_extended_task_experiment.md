# 08 — The extended task and its five-generation chain

16 September 2026. Local operational report; not tracked in Git.

> **Superseded, and kept as a record of its own stage.** The runs described here
> used the authors' 1e-05 on the extended task, before that task had been tuned.
> Stage 10 tuned it, selected 1e-3, and retired these runs to
> `temporary_checks/extended_task_at_learning_rate_1e-05/`. The measurements
> below are what stage 08 produced and are left unchanged; the paths it names
> have moved, and its conclusions are superseded by
> `findings/05_label_generation_strategies.md`.

## Files and functions I used, and what they gave me

| Source | What it does | What I took from it |
| --- | --- | --- |
| `CLAUDE.md` | Working conventions | Minimality rule, folder responsibilities, checkpoint policy, output policy, agent-output limits |
| `Development/reports/07_...md` | Previous stage | That the base chain produced five identical models, and why: a parent that answers everything correctly returns the real task to its successor |
| `upstream/models.py` `SequenceClassifier.call_with_all_aux` (:700-770) | Builds the token sequence and runs the transformer | **The key discovery**: it interleaves N symbols with N labels and drops the last label, so 4 symbols + 4 labels gives exactly the 7-token extended sequence. It also returns per-position hidden states and attention scores |
| `upstream/models.py` `AttentionBlock` (:260-360) | Multi-head attention | The fused `qkv` projection layout `(N, 3, heads, head_dim)`, which is what made weight-space ablation of one head's values straightforward |
| `upstream/main.py` `ce` (:59) | Cross-entropy | Reused unchanged for both label losses and the symbol loss |
| `upstream/main_utils.py` `get_model_from_opts` (:222), `get_optimizer_from_opts` (:250) | Model and optimizer construction | Backbone built exactly as the base task builds it, same constant-rate Adam |
| `upstream/samplers.py` `get_constant_burst_seq_idxs`, `get_mixed_seq_idxs`, `get_exemplar_inds`, `fewshot_relabel` | Original task generation | The opening contexts and queries, unchanged |
| `scripts/common.py` | Shared helpers | Paths, `baseline_options()`, `load_features()`, `training_seeds()`, `use_above_normal_priority()` |
| `scripts/base_task/analyse_runs.py` | Base-task analysis | The previous-token and induction measure definitions, reimplemented against the 7-token sequence |

## Files created, changed and removed

**Created — `scripts/extended_task/`:**

| File | Concrete purpose |
| --- | --- |
| `extended_model.py` | The model and its losses. Wraps the authors' backbone with one `Linear(64 → 2)` symbol head; provides teacher-forced losses, teacher-forced evaluation, and autoregressive continuation generation. |
| `run_extended.py` | The whole chain in one pipeline: builds each generation's dataset (correct for generation 0, parent-generated for successors), trains it, saves 55 checkpoints, `log.h5`, `config.json`, `training_data.h5` and `dataset_quality.json`. `--generations N` sets how many successors. |
| `analyse_extended.py` | Per generation: development scores at all three positions, attention measures, single-head ablations, within-training trajectory. Then the chain table and figure. |

**Created — `findings/03_extended_task_recursion.md`** and
`results/extended_task/` (five generation folders plus the comparison).

**Changed:** `README.md` (both tasks explained, extended commands, new outputs,
the untuned-rate limitation), `CLAUDE.md` (extended-task folder, and
`temporary_checks/`/`CLAUDE.md` now local-only),
`findings/01_learning_rate_search.md` (the 96.6% figure now reads as a strong
preference for context labels, not exclusive prediction of them).

**Removed:** a temporary correctness check under `Development/outputs/`, once it
had served its purpose. No timing files, verification reports or duplicate
evidence were created.

## Structure and tracking

A clone receives `README.md`, `.gitignore`, both requirements files, `scripts/`
(base and extended), `findings/`, `results/evaluation_data/class_splits.json`
and the vendored `upstream/`. Local-only and ignored: `Development/`,
`temporary_checks/`, `CLAUDE.md` and the rest of `results/`. `findings/`
tracking is unchanged. Nothing was committed, pushed or reset.

## Implementation decisions, and where they came from

**The 7-token sequence needed no new plumbing.** The backbone already builds
`sym, lab, sym, lab, …` and discards the final label. Feeding 4 symbols and 4
labels therefore produces
`symA | labA | symB | labB | querySym | queryLab | nextSym`, with the query label
appearing as a real input token at position 5 — which is exactly what teacher
forcing requires. Reading positions 4 and 6 of the existing label head gives both
label predictions for free.

**Only the symbol head is new.** It reads the hidden state at position 5 and
chooses between context position 0 and 1. It is an output head, not an attention
head, and it has its own initialisation key (seed 11) so adding it cannot disturb
the backbone's initialisation.

**Rotary positions made length irrelevant.** With `pos_embedding_type='rope'`
upstream sets the additive positional embedding to `Zeros()`, so going from five
to seven tokens needs no architectural change. A learned absolute embedding would
have had a fixed input width and would have needed changing.

**Ablation is done in weight space here.** The base task used upstream's `opto`
cache machinery, which is reached through their forward wrapper; the extended
model calls `call_with_all_aux` directly. Zeroing the value rows of the fused
`qkv` weight for one head (`start = 2*width + head*head_dim`) produces the same
intervention — that head writes nothing into the residual stream — and is easier
to explain.

**Sampling rule.** Query label and following label are argmax; the next symbol is
sampled at temperature 1, because it is the one output whose target is genuinely
random and argmax would collapse it to a constant.

## Checks performed before training

Run once in `Development/`, then deleted:

| Check | Result |
| --- | --- |
| Sequence length and layout | 7 tokens, as intended |
| Causal masking | Upper triangle of every attention matrix is zero |
| **No target leakage** | Changing the query-label token leaves the position-4 prediction bit-identical, while positions 5 and 6 react |
| Conditioning is correct | Changing the next symbol leaves both earlier predictions unchanged and only moves position 6 |
| Loss construction | Three finite components, mean equals their average |
| Opening contexts | First batch reproduces the base task exactly: classes `[13,44,13]`, labels `[2,4,2]` |
| Successor path | Generated datasets differ from the truth and use both context positions |

That last check caught a real bug. The base task wraps its sampler in
`get_mixed_seq_idxs` even with a single substrate, and that wrapper consumes a
random key. Calling the burst sampler directly produced *different* opening
questions. Adding the wrapper restored the match.

## Method

Learning rate 1e-5 (the authors' original), initialisation seed 5, batch size 32,
1,000,000 examples, 31,250 updates, fresh weights and optimizer per generation,
55 checkpoints written directly. Generation 0 learns correct continuations; each
successor learns continuations its parent generated autoregressively, mistakes
kept, nothing corrected. Scored on the fixed 1,000-question development
evaluator plus a reproducible coin flip (seed 13) held constant across
generations. The reserved final test was not generated or scored.

## Results

All five generations completed; every loss stayed finite.

| Gen | Query acc | Symbol loss | Next-label acc | Induction (L1) | Prev-token (L0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9620 | 0.6967 | 0.9490 | 0.7722 | 0.8086 |
| 1 | 0.9630 | 0.6980 | 0.9440 | 0.7662 | 0.8240 |
| 2 | 0.9610 | 0.6984 | 0.9440 | 0.7633 | 0.8273 |
| 3 | 0.9590 | 0.6987 | 0.9460 | 0.7596 | 0.8286 |
| 4 | 0.9570 | 0.6993 | 0.9430 | 0.7554 | 0.8313 |

Generated training data, per million examples:

| Dataset for gen | Query errors | Next-label errors | Chose position 0 |
| --- | ---: | ---: | ---: |
| 0 | 0.0000 | 0.0000 | 0.4994 |
| 1 | 0.0000 | 0.0000 | 0.4843 |
| 2 | 0.0000 | 0.0001 | 0.4829 |
| 3 | 0.0001 | 0.0002 | 0.4821 |
| 4 | 0.0001 | 0.0002 | 0.4746 |

## Interpretation

The chain drifted slightly and did not collapse. Query accuracy fell 0.5
percentage points across four generations — 5 questions out of the same fixed
1,000 — and the move was not monotone at the start. No test of whether that is
distinguishable from chance variation was carried out.

The substantive signal is the **symbol drift**, measured on a million generated
examples: the proportion choosing context position 0 moved monotonically from
0.4994 to 0.4746. All 50 classes still appear as next symbols at every
generation, which establishes retained coverage but not unchanged frequency
balance — relative class frequencies were not measured. Position preference and
symbol-identity preference are separate measurements; the position proportion is
what moved, and reporting only aggregate class counts would have hidden it.

The two attention measures moved in opposite directions (induction down, previous
token up) and the measured effect of silencing the induction head shrank from
−6.5 to −5.5 percentage points. All small; all attention-pattern measurements and
single-head intervention effects, which do not establish the circuit's overall
contribution.

Nothing here shows ordering: query accuracy and the induction score both drift
downwards over the same generations.

## Comparison with the base-task chain

The base chain at 1e-3 produced five models whose final parameters were equal
wherever we compared them; this one at 1e-5 produces slow drift. **Those two
experiments differ in both task and learning rate**, so the difference cannot be
attributed to the extended task alone. What the extended task adds is a quantity
the parent generates freely — the symbol choice — where a small bias could
compound. That the drift came from that route is a proposed explanation, not
something this stage tested.

## Remaining issues

1. **The extended task is untuned.** Whether 1e-5 or some other rate suits it is
   unknown; no tuning was done or is implied.
2. **One chain.** Single initialisation seed, single generation seed, single
   sampling rule.
3. **The drift is small.** Four generations of monotone movement in one measured
   quantity is suggestive; whether it continues, accelerates or flattens is
   untested, and generation 4 was the agreed stopping point.
