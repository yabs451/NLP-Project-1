# Finding 03 — Extended-task recursion: argmax labels hold, temperature-3 sampled labels collapse

16 September 2026 | source runs: `results/extended_task/`

## Question

How does the way a parent turns its label logits into training targets affect
performance and the induction-circuit measures across recursive generations?

Two chains of five models each share one generation 0 and differ **only** in
label generation:

- **`label_argmax`** — each generated label is the single most likely one.
- **`label_sampling_temperature_3`** — each generated label is drawn from
  `softmax(logits / 3)`, so the parent often writes down a label it did not
  think most likely. All five labels stay eligible.

We are also interested in whether measured circuit changes appear before
accuracy declines. Neither question required us to obtain collapse.

## The task

Each sequence opens exactly as the base task does — two symbol–label pairs and a
query symbol matching one of them, built by the authors' original generator. The
model then produces **three** outputs in turn:

1. the query's label,
2. a **next symbol**, chosen from the two symbols already in the context,
3. that symbol's label.

For a context `A → 1, B → 4` with query `A`, a valid continuation is `1, B, 4`.
Both label predictions have exactly one right answer; **either** symbol choice is
legitimate, so the symbol target is a fair coin flip and the best a model can do
is learn that distribution.

Restricting the next symbol to the two context symbols is **our design choice**,
not a requirement of the task definition.

### Model

The authors' two-layer attention-only backbone and their ResNet18 symbol
features are unchanged. Feeding four symbols and four labels into their existing
interleaving produces exactly the sequence the task needs:

```
0 sym A | 1 lab A | 2 sym B | 3 lab B | 4 query sym | 5 query lab | 6 next sym
```

Reading position 4 gives the query label, position 5 the next symbol, position 6
the following label. The backbone's own label head is reused for both label
predictions, so **the only new parameters are a two-way `Linear(64 → 2)` symbol
head** reading the hidden state at position 5. It is an output head, not an extra
attention head. Rotary positions make the move from five to seven tokens need no
architectural change.

Training is teacher-forced: the correct earlier continuation tokens are supplied,
cross-entropy is taken at the three output positions, and the objective is their
equally weighted mean. Attention is causal, so no prediction can see its own
target — verified by checking that changing the query-label token leaves the
position-4 prediction bit-identical while positions 5 and 6 react.

### How a successor's training data is made

The parent generates one token at a time, each step conditioned on what it
produced before, mistakes kept and never corrected:

1. the **query label** — argmax, or sampled from `softmax(logits / 3)`;
2. the **next symbol** — always sampled from the two-way head at temperature 1,
   in both conditions;
3. the **following label** — by the same rule as step 1.

Temperature divides the **logits** before the softmax; no already-normalised
probability is rescaled. Because generation is autoregressive, a sampled query
label is fed back in before the symbol is chosen, so the symbol probabilities
themselves differ between conditions — the symbol rule is identical but its
inputs are not.

Temperature applies **only** to making a successor's training data. The training
loss, the evaluation decoding and generation 0's data are unchanged.

Opening contexts always come from the original generator in both conditions, so
this is recursion over the *continuations* rather than fully synthetic generation.

Every generation: learning rate **1e-5** (the authors' original rate),
initialisation seed 5, batch size 32, 1,000,000 examples, 31,250 updates, fresh
weights and a fresh optimizer, 55 checkpoints. Generation 0 is trained once on
correct continuations and shared by both chains. Both are scored on the same
fixed 1,000-question development evaluator with the same decoding.

## Results

![condition comparison](../results/extended_task/condition_comparison.png)

### What the parents actually generated

Wrong labels among the million examples each successor trained on, measured
against the original context mapping:

| Dataset for gen | argmax: query / next | temperature 3: query / next |
| --- | ---: | ---: |
| 1 | 19 / 41 | **147,465 / 155,856** |
| 2 | 43 / 134 | **570,974 / 582,900** |
| 3 | 78 / 175 | **763,252 / 767,131** |
| 4 | 117 / 224 | **789,254 / 790,321** |

Argmax targets stay almost correct but accumulate: 19 → 117 wrong query labels,
a sixfold rise. Temperature-3 targets are wrong from the start — 15% at
generation 1 — and reach 79% by generation 4, approaching what uniform guessing
over five labels would give.

Both conditions keep all 50 training classes appearing as next symbols at every
generation, and neither shows a large shift in which context position is chosen.
The generation-0 dataset, whose symbols come from fair coin flips, sits at
0.4994; the argmax chain's generated datasets drift 0.4843 → 0.4746, while the
temperature-3 ones stay between 0.4842 and 0.4910 without a trend.

### What the trained models do

Teacher-forced development scores at each final checkpoint, identical evaluation
for both conditions:

| Gen | argmax: query acc / loss | T3: query acc / loss | argmax next acc | T3 next acc |
| --- | ---: | ---: | ---: | ---: |
| 0 (shared) | 0.962 / 0.104 | 0.962 / 0.104 | 0.949 | 0.949 |
| 1 | 0.963 / 0.100 | 0.951 / 0.279 | 0.944 | 0.941 |
| 2 | 0.961 / 0.099 | **0.601 / 1.182** | 0.944 | **0.546** |
| 3 | 0.959 / 0.103 | **0.501 / 1.462** | 0.946 | **0.457** |
| 4 | 0.957 / 0.105 | **0.366 / 1.561** | 0.943 | **0.351** |

Symbol loss stays near ln 2 ≈ 0.6931 in both conditions (argmax 0.6967–0.6993,
slightly above; temperature 3 0.6924–0.7001), so neither chain lost the coin-flip
behaviour on the symbol output even as the labels collapsed.

### Attention measures and ablations

| Gen | argmax induction / prev-token | T3 induction / prev-token |
| --- | ---: | ---: |
| 0 | 0.7722 / 0.8086 | 0.7722 / 0.8086 |
| 1 | 0.7662 / 0.8240 | 0.6922 / 0.8175 |
| 2 | 0.7633 / 0.8273 | **0.0959 / 0.2163** |
| 3 | 0.7596 / 0.8286 | **0.0036 / 0.1050** |
| 4 | 0.7554 / 0.8313 | **0.0041 / 0.0985** |

Heads were selected from each model's own final scores, never inherited. In the
argmax chain the same heads came out every time (induction L1H2, previous-token
L0H5). **In the temperature-3 chain the selected head identity changed** — L1H2,
then L1H3, then L1H4 — so its ablation numbers across generations refer to
different heads and are not a like-for-like series.

Zeroing the selected induction head's value vectors costs 6.5 pp of query
accuracy at generation 0 and 5.5 pp at argmax generation 4. In the temperature-3
chain the cost stays in the 5.6–8.4 pp range even at generation 4, where intact
accuracy is only 0.366 — a similar absolute cost from a much lower base.

12 of the 55 saved checkpoints were analysed per generation, by the same rule in
both conditions.

## Interpretation

**Label generation strategy decided the outcome.** With argmax labels the chain
held: query accuracy moved 0.962 → 0.957 over four generations. With
temperature-3 sampled labels it fell to 0.366, moving towards — but still above
— the 0.2 that uniform guessing over five labels would give. This is the
clearest degradation the project has produced.

**The mechanism is visible in the training data.** At temperature 3 the parent
writes down a label it did not consider most likely a large fraction of the time;
those labels become the successor's targets, the successor learns them, and its
own outputs are worse still. The error rate compounds 15% → 57% → 76% → 79%.

**On the ordering question, generation 1 is the only informative point, and it is
suggestive rather than conclusive.** Going from generation 0 to temperature-3
generation 1, query accuracy fell 1.1% relative (0.962 → 0.951) while the
strongest induction score fell 10.4% relative (0.7722 → 0.6922). By generation 2
both had moved sharply. So the circuit measure moved proportionally more than
accuracy at the first step. Three cautions: the two quantities are on different
scales and a percentage comparison between them is not a like-for-like test;
query *loss* also rose steeply at generation 1 (0.104 → 0.279), so it is not the
case that behaviour was intact while only the mechanism changed; and this is one
generation in one chain.

**A drop in an attention score is not a demonstrated loss of circuit function.**
The induction score falling to ~0.004 says the layer-1 heads no longer attend
preferentially to the matching label token. That is consistent with the circuit
having stopped operating, but the ablation results do not confirm it
independently: silencing one head still costs a similar number of accuracy
points as it did at generation 0. Single-head ablation measures one
intervention, not the importance of the circuit as a whole.

**The symbol output survived.** Both chains keep symbol loss near ln 2 and keep
all 50 classes appearing. The collapse is specific to the label outputs, which
are the outputs whose targets were corrupted.

## Limitations

- **This comparison is argmax versus temperature-3 sampling.** It does not
  isolate temperature: sampling at temperature 1 is a separate condition and was
  not run. Nothing here establishes a temperature-only effect or a general law.
- **One chain per condition**, one initialisation seed, one training budget, one
  development set, one learning rate (1e-5, untuned for this task).
- **Head identity changed across the temperature-3 chain**, so its ablation
  series is not a like-for-like comparison.
- All 50 classes appearing establishes retained coverage, not unchanged relative
  frequencies, which were not measured.
- 12 of 55 checkpoints analysed per generation; the within-training trajectories
  in each `analysis.json` are at those points only.
- The reserved final-test classes have never been scored.

## Reproduce

From the project root, with the environment set up (see README):

```powershell
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --generations 4
.\.venv\Scripts\python.exe scripts/extended_task/run_extended.py --label-strategy sample --label-temperature 3 --generations 4
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition label_argmax
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --condition label_sampling_temperature_3
.\.venv\Scripts\python.exe scripts/extended_task/analyse_extended.py --compare-conditions
```

The first command trains the shared generation 0 and the argmax chain; the
second reuses that generation 0. Finished generations are skipped.

Measurements and figures:

- `results/extended_task/condition_comparison.json` and `.png` — the two
  conditions side by side
- `results/extended_task/experiments/<condition>/generation_comparison.json` and
  `.png` — one chain's table and figure
- `results/extended_task/experiments/<condition>/generation_<n>/analysis.json` —
  per-generation development scores, attention measures, ablations and the
  within-training trajectory
- `.../generation_<n>/dataset_quality.json` — what the parent generated, as
  counts and rates
- `results/extended_task/generation_0/` — the shared parent

Seeds are recorded in each generation's `config.json`: symbol-head
initialisation 11, generation-0 coin flips 12, development coin flips 13,
next-symbol sampling 14, label sampling 15.
