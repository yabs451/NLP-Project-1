# 11 — Two symbol-side experiment families: context feedback and next-symbol temperature

18 September 2026. Local operational report; not tracked in Git.

## Files I read, and what I took from them

| Source | Role | What I took from it |
| --- | --- | --- |
| `CLAUDE.md` | Working conventions | Minimality rule, folder responsibilities, checkpoint policy, outputs to keep and not keep, agent-output limits |
| `Development/reports/10_...md` | Previous stage | The tuned-rate setup, the shared generation 0, the corrections standards and the existing distribution analysis |
| `findings/05_label_generation_strategies.md` | The label family | The reference condition and the measurements the new families had to line up with |
| `upstream/icl-dynamics/samplers.py` `get_constant_burst_seq_idxs` (:32–120) | The question generator | **The key detail**: `class_distr` is a probability vector over the training classes. The query class is drawn from it, then the distractor from the same weights with the query masked out and renormalised — so passing weights straight in gives exactly the "two distinct classes, sampled without replacement" rule the design needs, with no change to label pairs, query construction or exemplars |
| `scripts/extended_task/run_extended.py` | The chain pipeline | `draw_opening_contexts`, `build_successor_dataset`, `train_generation`, `write_config` — all extended rather than duplicated |
| `scripts/extended_task/extended_model.py` `generate_continuation` | Autoregressive generation | Where the next-symbol temperature had to be applied, before the categorical draw |
| `scripts/extended_task/analyse_extended.py` | Analysis | `symbol_breakdown`, `dataset_distributions`, `plot_conditions` — extended for the new stages and families |
| `results/extended_task/recursive/generation_0/` | Shared parent | Reused unchanged; not retrained, and no tuning repeated |

I verified the class-index convention rather than assuming it: the 50 training
classes are global ids 0–49, so a weight vector in `splits["class"]["train"]`
order indexes directly, and per-class counts are reordered through that list
rather than relying on the coincidence.

## Files changed and created

| File | Change | Why |
| --- | --- | --- |
| `scripts/extended_task/extended_model.py` | `generate_continuation` takes `symbol_temperature`, applied to the two next-symbol logits before the categorical draw | Family 2's only mechanism |
| `scripts/extended_task/run_extended.py` | `draw_opening_contexts` accepts `class_weights`; new `measure_generated_symbol_frequencies` and `context_sampling_weights`; `build_successor_dataset` threads both new settings; `write_config` records them plus the `context_feedback` block; `condition_name` covers three families; `temperature_value` parses `1/3`; `--next-symbol-temperature` and `--context-temperature` | One pipeline runs all nine conditions. No separate training script for any condition |
| `scripts/extended_task/analyse_extended.py` | `symbol_breakdown` now returns per-class offers, selections and selection rates plus offered-share and offered-entropy; `dataset_distributions` carries the two config-side feedback stages; new `plot_symbol_distributions`; `--compare-conditions` became `--compare-family {label,context,symbol}` | The four stages had to stay distinguishable, and nine conditions on one chart would have been unreadable |
| `scripts/common.py` | Fixed `use_above_normal_priority` | See below — it had never worked |
| `findings/06_symbol_distribution_experiments.md` | Created | The scientific record for both new families |
| `findings/05_label_generation_strategies.md`, `README.md` | Command renamed to `--compare-family label`; README rewritten for three families, nine conditions, new commands and output paths | The flag changed, so every active reference had to |

New outputs, all under `results/extended_task/recursive/`: five condition folders
with four generations each, `context_feedback_comparison.json`/`.png`,
`symbol_sampling_comparison.json`/`.png`,
`context_feedback_distributions.png`, `symbol_sampling_distributions.png`.
Existing label-family results, `condition_comparison.*` and
`data_distributions.png`, keep their names and content.

Nothing was created outside these. No timing files, runtime columns, monitoring
outputs or permanent check utilities. Both requirements files, `upstream/`, the
base-task experiments and the archived experiment are untouched.

## Structure

```
results/extended_task/recursive/
  generation_0/                                  shared by all nine conditions
  experiments/
    label_argmax/                                the shared reference
    label_sampling_temperature_1 | _3 | _5       family 1 (finding 05)
    context_feedback_temperature_1               family 2
    context_feedback_temperature_one_third
    context_feedback_temperature_0.2
    symbol_sampling_temperature_one_third        family 3
    symbol_sampling_temperature_0.2
  condition_comparison.*        data_distributions.png            family 1
  context_feedback_comparison.* context_feedback_distributions.png family 2
  symbol_sampling_comparison.*  symbol_sampling_distributions.png  family 3
```

`Development/`, `temporary_checks/` and `CLAUDE.md` remain local and ignored.
Nothing was committed, pushed, reset or staged.

## Checks before running

One temporary script in `Development/outputs/`, deleted afterwards, confirmed:

- **The reference condition is unchanged.** Regenerating `label_argmax`
  generation 1 through the refactored code reproduced the saved openings and all
  three generated outputs **bit-identically**, so adding two settings did not
  disturb the completed work.
- Colder symbol sampling moves the position split in the expected direction on a
  2,000-example probe (0.484 → 0.459 → 0.436 at temperatures 1, 1/3, 0.2).
- The frequency pass, the weighting rule and weighted question construction all
  behave: weights normalise, the two context classes stay distinct in every row,
  the query is always one of them, and the offered frequencies track the supplied
  weights without matching them exactly.

## A real defect found: process priority had never been raised

While checking on the running job I found the training process at **Normal**
priority. `common.use_above_normal_priority()` called
`SetPriorityClass(GetCurrentProcess(), ...)` without declaring the signature.
`GetCurrentProcess()` returns the pseudo-handle −1, which ctypes passed as a
32-bit int; Windows needs a full-width handle, so the call received an invalid
one and failed — and the helper ignored the return value, so it failed silently.
Reproduced in a clean process: priority class stayed `0x20` after the call.

This affected every run that used the helper, in this session and earlier ones.
It changes scheduling, not computation, so no scientific result is affected.

Fixed by declaring `GetCurrentProcess.restype` and `SetPriorityClass.argtypes`,
using a private `WinDLL` handle so it cannot disturb another module's ctypes
state, and printing one line if Windows refuses instead of failing silently.
Verified: the call now returns `0x8000`. The running job was raised externally,
and the later conditions — fresh processes — elevated themselves.

Incidentally, `.venv\Scripts\python.exe` on this machine is a launcher stub that
spawns the base interpreter as a child, so there are always two python PIDs; the
worker is the one with the threads and CPU time.

## Results

20 successors trained, five conditions, run sequentially. All completed.

**Context feedback.** Largest single-class share by stage at generation 4:

| condition | weights supplied | offered | selected | classes selected | entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | — | 0.0203 | 0.0203 | 50 | 3.9120 |
| T = 1 | 0.0215 | 0.0216 | 0.0216 | 50 | 3.9115 |
| T = 1/3 | 0.1291 | 0.1221 | 0.1216 | 50 | 3.6071 |
| T = 0.2 | 0.9967 | 0.5000 | 0.5001 | 40 | 1.8404 |

Feedback without sharpening is nearly inert. The fifth power reaches one class
holding 99.7% of the weight, but the distinct-class rule caps what that class can
*be* — offered in nearly every question, selected in half of them, with 39 others
still appearing. The stop condition (fewer than two classes with positive weight)
never triggered.

Performance held at temperatures 1 and 1/3 throughout (0.992–0.999 query
accuracy) and at temperature 0.2 for three generations, then generation 4 fell to
0.535 with query loss 5.551, following-label accuracy 0.497 and induction 0.161 —
the only run in this family that never crossed 90% at any logged point. **Its
training labels were 100% correct**, so that collapse came from the questions,
not from corrupted targets.

**Next-symbol temperature.** Share choosing context position 0:

| condition | gen 1 | gen 2 | gen 3 | gen 4 |
| --- | ---: | ---: | ---: | ---: |
| reference | 0.4879 | 0.4943 | 0.4958 | 0.5015 |
| T = 1/3 | 0.4624 | 0.3948 | 0.2345 | 0.1358 |
| T = 0.2 | 0.4379 | 0.2556 | 0.1204 | 0.1231 |

Identity barely moved: offered share fixed at 0.02026 (openings unchanged),
largest selected share 0.0208, entropy 3.9119 against the reference's 3.9120, all
50 classes selected every time. The availability-adjusted identity measure rose
from 0.0022 to 0.0068–0.0079, about three times the coin-flip floor, without
producing a concentrated distribution.

Label outputs stayed intact (query accuracy 0.991–0.995) while **symbol loss rose
from ln 2 ≈ 0.693 to 1.95 and 4.38 nats** — the model committing confidently to
one position against a target that is a fair coin flip on the original evaluator.
These chains also crossed 90% query accuracy earlier each generation (60,000 →
25,024 at T = 1/3).

## Limitations

1. **One chain per condition** from one shared parent, one seed, one budget. Not
   replications.
2. **The sharpening is an imposed intervention**, not a model preference; the
   amplification comes from the exponent we chose.
3. **The evaluator is deliberately the original distribution.** For the
   context-feedback conditions that means training and evaluation distributions
   diverge, so those scores measure the original questions only; the models were
   not scored on their own training distribution.
4. **The context-family collapse is one generation in one chain**, so the link
   between extreme concentration and collapse rests on a single point.
5. **The availability-adjusted identity measure is unreliable under extreme
   imbalance** — it reads 0.164 at context temperature 0.2 generation 4, which is
   noise from rarely-offered classes rather than a preference.
6. **Why the previous-token ablation effect ranges from under 1 to 41 percentage
   points** across these runs is unexplained and recorded, not interpreted.
7. **Why the symbol-temperature chains reach 90% earlier each generation** is
   untested; a more deterministic target being easier to fit is a guess.
8. Attention coverage is 12 of 55 checkpoints; head identities move between
   generations in most of these chains.

## Extension to generation 6, and a wording correction pass (18 September 2026)

### What ran

The four sharpened symbol chains were continued from their own saved generation-4
models through generations 5 and 6 — eight new successors. The reference and
context temperature 1 were **not** extended, and generations 0–4 were not
retrained.

**The generation-count option names the final generation, not an increment.** I
checked the loop in `run_extended.py` before running: it iterates `1 .. N`,
loading any generation that already has a `log.h5` as the parent and training
only the missing ones. So `--generations 6` extends a finished four-generation
chain by exactly two, and each chain continued from its own evolving
distribution — for context feedback, `parent_folder` is the skipped generation-4
folder, so the frequency pass reads that chain's own generation-4 dataset.

No condition stopped under the two-distinct-classes rule. All four reached
generation 6.

### Files used and changed

| File | Role | Change |
| --- | --- | --- |
| `scripts/extended_task/run_extended.py` | The chain pipeline | **Unchanged.** Its resume path already did exactly what an extension needs |
| `scripts/extended_task/analyse_extended.py` | Analysis and figures | One fix: `plot_conditions`, `plot_distributions` and `plot_symbol_distributions` took their x-ticks from a single condition, which would have truncated the axis at generation 4 once chains differed in length. They now use the union of generations present |
| `findings/06_symbol_distribution_experiments.md` | The scientific record | Rewritten for generations 0–6 and for the wording corrections below |
| `findings/05_label_generation_strategies.md` | Label family | "Collapsed" now says which measured behaviour fell, with a one-line definition at first use |
| `README.md` | Public guide | `--generations 6` for the four extended chains, plus a note that the option names the final generation and that chains deliberately differ in length |

No new script, folder or figure was created; the existing condition folders and
family outputs were extended in place.

### Results

**Context feedback.** Extending changed the earlier conclusion. Temperature 1/3
looked stable at generation 4 (0.997 query accuracy) and then fell to 0.974 at
generation 5 and **0.410** at generation 6. Temperature 0.2, which had fallen to
0.535 at generation 4, did not keep falling: 0.615 at generation 5, 0.526 at
generation 6 — a fluctuating plateau. Symbol coverage kept shrinking, from 50
distinct classes selected to 34 (T = 1/3) and to 5 (T = 0.2) by generation 6.

Sampling weights concentrated to 0.9905 and 0.8473, while the **offered** share
cannot exceed 0.5 because the two context classes must differ, and the
**selected** share was measured near 0.5000 throughout — the latter is a
measurement, not a cap, since a parent that always picked the dominant slot could
have generated it far more often. Wrong query labels across the temperature-1/3 chain were
exactly 0 in all six datasets.

**Next-symbol temperature.** Positional preference **plateaued** near 0.12 rather
than continuing toward 0 (T = 1/3: 0.136 → 0.115 → 0.116; T = 0.2: 0.122 → 0.120
→ 0.120). Symbol-identity concentration did not move at all in six generations.
Symbol loss kept rising, to 4.13 and 5.75 nats. Following-label accuracy dipped to
0.898 at temperature 0.2 generation 4 and recovered to 0.940. Query-label
accuracy never degraded and ended at 0.997.

The clearest dissociation in the project appeared here: along the temperature-0.2
chain the induction score fell 0.977 → 0.646 while query-label accuracy rose to
its highest value in that chain.

### Wording corrected

| Claim | Correction |
| --- | --- |
| "two ways to break a chain without corrupting a single label" (finding 06 title) | Removed. Several datasets in these families did contain wrong labels — up to 618 per million at context temperature 1 generation 3 |
| "The label outputs are essentially untouched" (symbol family) | Wrong. Query-label accuracy held, but **following-label accuracy fell to 0.898** at temperature 0.2 generation 4. The two label outputs are now reported separately |
| "the generated labels stayed essentially perfect … exactly zero in most datasets" | Replaced by exact per-dataset counts, naming temperature 1/3 as the one chain where all six datasets had zero wrong labels |
| "one class holding almost all the sampling weight" read as a symbol frequency | Now states explicitly that 0.9967 of the **sampling weight** produced 0.5000 of the **offered** slots and 0.5001 of the **selected** symbols |
| classes "no longer appearing" | Now distinguishes absence from a finite million-example sample — which this rule turns into exactly zero weight and hence permanent removal from that chain — from a claim that the model assigns zero probability, which was never measured |
| "collapse" used unqualified | Every use now names the measured behaviour: query-label accuracy, following-label accuracy, symbol loss, identity concentration, coverage or positional preference |
| concentration presented as the explanation | Now stated as co-occurrence only; the correct labels rule corrupted targets out as an explanation here, but nothing separates concentration from other factors |
| attention scores read as circuit function | Reinforced with the concrete counter-example above |
| "stable" claims | Now scoped to the generations actually run, with the temperature-1/3 reversal given as the reason that scoping matters |

### Correction to an earlier operational statement in this report

The stage-11 section above reports that `use_above_normal_priority()` was fixed
and that later conditions elevated themselves. That remains accurate. Two things
to add: the CPU-priority material belongs only here and was never written into any
finding, and the extension runs used the fixed mechanism at launch with no manual
intervention — confirmed once, not polled.

### Limitations added by the extension

1. **Six generations is still a budget, not a limit.** Temperature 1/3 looked
   stable at four and had fallen sharply by six; nothing here says what
   generation 7 would show for any chain.
2. **The temperature-0.2 plateau is three points** (0.535, 0.615, 0.526) in one
   chain, so "plateau" describes what was observed, not an established
   equilibrium.
3. **The partial recovery at temperature 0.2 generation 5 is unexplained.**
4. **Chains differ in length**, so comparisons at generations 5 and 6 involve
   only the four extended conditions; the reference was not extended and is not
   extrapolated.
