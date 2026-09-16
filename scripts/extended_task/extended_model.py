"""The extended task: predict the query label, a next symbol, then its label.

The model keeps the authors' two-layer attention-only backbone and their symbol
features unchanged. The only addition is a two-way output head that picks which
of the two context symbols comes next.

Task and design decisions: findings/03_extended_task_recursion.md
"""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

import equinox as eqx
import equinox.nn as enn
import jax
import jax.numpy as jnp
import main as upstream_main
import main_utils

# Token layout once a two-pair context is extended:
#   0 sym A | 1 lab A | 2 sym B | 3 lab B | 4 query sym | 5 query lab | 6 next sym
# The backbone interleaves 4 symbols with 4 labels and drops the last label, so
# feeding 4 symbols and 4 labels produces exactly these 7 tokens.
QUERY_LABEL_POSITION = 4     # reading here predicts the query's label
NEXT_SYMBOL_POSITION = 5     # reading here predicts which context symbol follows
NEXT_LABEL_POSITION = 6      # reading here predicts that symbol's label
NUM_SYMBOL_CHOICES = 2       # the next symbol is always one of the two in context


class ExtendedModel(eqx.Module):
    """The authors' classifier plus a two-way symbol-choice head.

    `backbone` is upstream's SequenceClassifier, untouched: its label head is
    reused for both label predictions. `symbol_head` is the only new parameter
    block, mapping one hidden state to a choice between context position 0 or 1.
    It is an output head, not an extra attention head.
    """
    backbone: eqx.Module
    symbol_head: enn.Linear

    def __init__(self, opts, key):
        self.backbone = main_utils.get_model_from_opts(opts)
        self.symbol_head = enn.Linear(opts.d_model, NUM_SYMBOL_CHOICES,
                                      use_bias=True, key=key)


def build_model(opts, symbol_head_seed):
    """Create a fresh extended model.

    The backbone is initialised from `opts.init_seed` exactly as the base task
    does; the symbol head gets its own key so adding it cannot disturb the
    backbone's initialisation. `get_model_from_opts` mutates the options it is
    given, so it receives a copy.
    """
    return ExtendedModel(copy.deepcopy(opts), jax.random.PRNGKey(symbol_head_seed))


def add_next_symbol(context_symbols, symbol_choice):
    """Append the chosen context symbol to the three opening symbols.

    context_symbols: [batch, 3, 512] — context A, context B, query.
    symbol_choice: [batch] — 0 or 1, which context symbol comes next.
    Returns [batch, 4, 512].
    """
    rows = jnp.arange(context_symbols.shape[0])
    chosen = context_symbols[rows, symbol_choice]
    return jnp.concatenate([context_symbols, chosen[:, None, :]], axis=1)


def add_label_slot(labels):
    """Pad the three label tokens with the slot the backbone discards.

    labels: [batch, 3] — context A, context B, and the query label fed at
    position 5 under teacher forcing. The fourth entry is never a target and is
    never visible to any prediction.
    """
    return jnp.concatenate([labels, jnp.zeros((labels.shape[0], 1), labels.dtype)], axis=1)


def forward(model, symbols, label_tokens, key):
    """Run one sequence and return the three prediction logits.

    symbols: [4, 512]; label_tokens: [4]. Returns query-label logits [5],
    symbol-choice logits [2] and next-label logits [5]. Attention is causal, so
    each prediction sees only tokens up to and including its own position.
    """
    aux = model.backbone.call_with_all_aux(examples=symbols, labels=label_tokens, key=key)
    label_logits = aux["unembedding"]                      # [7, 5]
    hidden = aux["transformer_output"]["out"]              # [7, d_model]
    return (label_logits[QUERY_LABEL_POSITION],
            model.symbol_head(hidden[NEXT_SYMBOL_POSITION]),
            label_logits[NEXT_LABEL_POSITION])


def batched_forward(model, context_symbols, labels, symbol_choice, key):
    """Vectorise `forward` over a batch, assembling the 7-token sequence first."""
    symbols = add_next_symbol(context_symbols, symbol_choice)
    label_tokens = add_label_slot(labels)
    keys = jax.random.split(key, symbols.shape[0])
    return jax.vmap(forward, in_axes=(None, 0, 0, 0))(model, symbols, label_tokens, keys)


def losses(model, context_symbols, labels, symbol_choice, next_label, key):
    """Cross-entropy at each of the three output positions, averaged over the batch.

    `labels[:, 2]` is the query-label target and also the token fed at position
    5. Returns (mean of the three, query loss, symbol loss, next-label loss),
    all in nats, the three weighted equally.
    """
    query_logits, symbol_logits, next_logits = batched_forward(
        model, context_symbols, labels, symbol_choice, key)
    query_loss = jnp.mean(upstream_main.ce(query_logits, labels[:, 2]))
    symbol_loss = jnp.mean(upstream_main.ce(symbol_logits, symbol_choice))
    next_loss = jnp.mean(upstream_main.ce(next_logits, next_label))
    return (query_loss + symbol_loss + next_loss) / 3, query_loss, symbol_loss, next_loss


@eqx.filter_jit
def train_step(model, optimizer, opt_state, context_symbols, labels, symbol_choice,
               next_label, key):
    """One Adam update on the equally weighted three-position objective."""
    def objective(model):
        total, query, symbol, following = losses(
            model, context_symbols, labels, symbol_choice, next_label, key)
        return total, (query, symbol, following)

    (total, parts), grads = eqx.filter_value_and_grad(objective, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    return eqx.apply_updates(model, updates), opt_state, total, parts


@eqx.filter_jit
def evaluate_teacher_forced(model, context_symbols, labels, symbol_choice,
                            next_label, true_next_label, key):
    """Score a batch with the correct preceding continuation tokens supplied.

    Each prediction is conditioned on the true earlier tokens, so the three
    positions are measured independently. `true_next_label` is the correct label
    of the chosen symbol under the original context mapping.
    """
    query_logits, symbol_logits, next_logits = batched_forward(
        model, context_symbols, labels, symbol_choice, key)
    return {
        "query_label_accuracy": jnp.mean(jnp.argmax(query_logits, -1) == labels[:, 2]),
        "query_label_loss": jnp.mean(upstream_main.ce(query_logits, labels[:, 2])),
        "symbol_loss": jnp.mean(upstream_main.ce(symbol_logits, symbol_choice)),
        "symbol_matches_target": jnp.mean(jnp.argmax(symbol_logits, -1) == symbol_choice),
        # How often the head prefers context position 0, whatever the target was.
        "symbol_prefers_first": jnp.mean(jnp.argmax(symbol_logits, -1) == 0),
        "next_label_accuracy": jnp.mean(jnp.argmax(next_logits, -1) == true_next_label),
        "next_label_loss": jnp.mean(upstream_main.ce(next_logits, true_next_label)),
    }


def choose_label(logits, strategy, temperature, key):
    """Turn label logits into a label, by argmax or by temperature sampling.

    Sampling divides the *logits* by the temperature before the softmax, which
    is what `categorical` does internally, so no already-normalised probability
    is ever rescaled. All five labels stay eligible.
    """
    if strategy == "argmax":
        return jnp.argmax(logits, axis=-1)
    return jax.random.categorical(key, logits / temperature, axis=-1)


@eqx.filter_jit
def generate_continuation(model, context_symbols, context_labels, key, label_key,
                          label_strategy="argmax", label_temperature=1.0):
    """Produce a continuation the way a parent model does, one token at a time.

    Each step conditions on the tokens the model itself produced, mistakes
    included. The next symbol is always sampled from the two-way head at
    temperature 1; the two labels follow `label_strategy`.

    `key` drives the symbol sampling and `label_key` the label sampling, kept
    separate so that adding label sampling leaves the symbol stream unchanged.

    Because generation is autoregressive, a sampled query label is fed back in
    before the symbol is chosen, so the symbol probabilities themselves differ
    between label strategies.

    context_labels: [batch, 2] — the two correct context labels.
    Returns generated query label, symbol choice and next label, each [batch].
    """
    batch = context_symbols.shape[0]
    zeros = jnp.zeros((batch,), dtype=jnp.int32)
    query_key, next_key = jax.random.split(label_key)

    # Step 1: the query label. Causal masking means tokens after the query
    # symbol cannot influence it, so later slots may hold placeholders.
    stage1 = jnp.concatenate([context_labels, zeros[:, None]], axis=1)
    query_logits, _, _ = batched_forward(model, context_symbols, stage1, zeros, key)
    query_label = choose_label(query_logits, label_strategy, label_temperature, query_key)

    # Step 2: the next symbol, conditioned on the label just generated.
    stage2 = jnp.concatenate([context_labels, query_label[:, None]], axis=1)
    _, symbol_logits, _ = batched_forward(model, context_symbols, stage2, zeros, key)
    symbol_key, _ = jax.random.split(key)
    symbol_choice = jax.random.categorical(symbol_key, symbol_logits, axis=-1)

    # Step 3: the following label, conditioned on the symbol just sampled.
    _, _, next_logits = batched_forward(model, context_symbols, stage2, symbol_choice, key)
    return (query_label, symbol_choice,
            choose_label(next_logits, label_strategy, label_temperature, next_key))
