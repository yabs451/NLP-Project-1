"""Helpers shared by every script in this project.

Nothing here reimplements the authors' science. It only locates their code,
loads their checkpoints, and rebuilds the evaluator sets exactly as their
`main.py` does, so that our scripts and their scripts agree.

Import this before importing anything from `upstream/icl-dynamics/`: setting
`JAX_PLATFORMS` has to happen before JAX is first imported.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "icl-dynamics"
FEATURE_FILE = UPSTREAM / "omniglot_resnet18_randomized_order_s0.h5"
EVALUATION_DATA = ROOT / "results" / "evaluation_data"
# Two development evaluators, both on the same 100 held-out classes:
# the small one is loaded into training and scored at every evaluation point;
# the large one is never loaded into training and scores final checkpoints only.
DEV_EVALUATOR_FILE = EVALUATION_DATA / "eval_dev.h5"
LARGE_DEV_EVALUATOR_FILE = EVALUATION_DATA / "eval_dev_large.h5"

# JAX has no native-Windows GPU build, and we want runs to be comparable
# regardless of what hardware happens to be present.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.dont_write_bytecode = True
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))


def baseline_arguments():
    """The authors' own baseline command, read out of their shell script.

    We take the first `python main.py` line of `ih_paper_runs.sh` (the Figure 3a
    `omniglot50_rl5` run), substitute the shell variables the script expects,
    and repoint the data file at our vendored copy. Reading the arguments from
    their file rather than retyping them means we cannot silently drift from the
    published configuration.
    """
    source = (UPSTREAM / "ih_paper_runs.sh").read_text()
    line = next(line for line in source.splitlines() if line.startswith("python main.py "))
    for name, value in {"MAIN_RUN_ITERS": "1000000", "INIT_SEED": "5",
                        "SAVE_FOLDER": "./ih_paper_reprod/main_paper_is5_ih3_pt2"}.items():
        line = line.replace("$" + name, value)
    args = shlex.split(line)[2:]           # drop "python main.py"
    args[args.index("--data_file") + 1] = str(FEATURE_FILE)
    return args


def baseline_options():
    """Parse the authors' baseline command into a fully resolved options object."""
    import main_utils
    import opto
    parser = main_utils.create_parser()
    opto.add_args_to_parser(parser)
    opts = parser.parse_args(baseline_arguments())
    opts.train_microbs = opts.train_bs
    opts.model_output_classes = opts.fs_relabel
    main_utils.check_opts(opts)
    return opts


def load_run_options(run_folder):
    """Load the options a finished run actually used, from its own config.json."""
    return argparse.Namespace(**json.loads((Path(run_folder) / "config.json").read_text()))


def checkpoint_path(run_folder, iteration):
    """Checkpoints are named by sequence count, zero-padded to 11 digits."""
    return Path(run_folder) / "checkpoints" / "{:011d}.eqx".format(iteration)


def available_checkpoints(run_folder):
    return sorted(int(p.stem) for p in (Path(run_folder) / "checkpoints").glob("*.eqx"))


def fresh_model_and_optimizer(opts):
    """Build an untrained model and a fresh optimizer from the authors' code.

    The model is initialised from `opts.init_seed`, so calling this twice with
    the same options gives identical starting weights.
    """
    import main_utils
    # get_model_from_opts mutates what it is given, so pass it a copy.
    model = main_utils.get_model_from_opts(argparse.Namespace(**vars(opts)))
    optimizer = main_utils.get_optimizer_from_opts(opts)
    return model, optimizer


def load_checkpoint(run_folder, iteration, opts):
    """Deserialise one checkpoint (model, optimizer state and PRNG keys).

    Equinox needs a correctly shaped template to deserialise into, which is why
    we build a fresh model and optimizer state first.
    """
    import equinox as eqx
    import jax
    model, optimizer = fresh_model_and_optimizer(opts)
    template = {"iter": -1, "model": model,
                "opt_state": optimizer.init(eqx.filter(model, eqx.is_array)),
                "seeds": {name: jax.random.PRNGKey(0) for name in
                          ("eval_model_seed", "train_data_seed", "train_model_seed")}}
    checkpoint = eqx.tree_deserialise_leaves(checkpoint_path(run_folder, iteration), template)
    assert checkpoint["iter"] == iteration, "checkpoint reports a different iteration"
    return checkpoint


def load_features():
    """The (1623, 5, 512) array of precomputed Omniglot ResNet18 features.

    Axis 0 is the character class (a row position in this already-shuffled
    file), axis 1 is which of the five exemplars, axis 2 the feature vector.
    """
    import h5py
    import jax.numpy as jnp
    opts = baseline_options()
    with h5py.File(opts.data_file, "r") as handle:
        return jnp.asarray(handle[opts.data_path_in_file][:])


def build_evaluators(opts, features):
    """Rebuild a run's fixed evaluation sets exactly as main.py builds them.

    Mirrors upstream/icl-dynamics/main.py:356-400. Evaluators named in
    `opts.load_eval_data` files are read from disk first; the rest are sampled
    once from `opts.eval_seed` and then reused unchanged for the whole run, so
    every checkpoint is scored on identical questions.

    Returns {evaluator_name: {'examples': [n, 3, 512], 'labels': [n, 3]}}.
    """
    from functools import partial
    import h5py
    import jax
    import jax.numpy as jnp
    import main_utils
    import samplers
    from main import smart_index

    splits = main_utils.get_splits_from_opts(opts, features.shape)
    evaluators = {}
    for eval_file in (opts.load_eval_data or []):
        with h5py.File(eval_file, "r") as handle:
            for name in handle:
                if name in evaluators:
                    raise ValueError("duplicate evaluator name: " + name)
                evaluators[name] = {field: jnp.asarray(handle[name][field][:])
                                    for field in ("examples", "labels")}

    # main.py turns eval_seed into a key, splits it into (data, model) halves,
    # then splits the data half once per evaluator. Reproducing that chain
    # exactly is what makes our rebuilt evaluators identical to the run's own.
    eval_data_seed, _ = evaluation_seeds(opts)
    seeds = jax.random.split(eval_data_seed, len(opts.pe_names))
    for index, name in enumerate(opts.pe_names):
        classes = splits["class"][opts.pe_classes[index]]
        class_sampler = partial(
            samplers.get_constant_burst_seq_idxs, classes=classes,
            class_distr=jnp.ones(len(classes)) / len(classes), num_seqs=opts.eval_iters,
            context_len=smart_index(opts.pe_context_len, index, opts.train_context_len),
            burstiness=opts.pe_burstiness[index],
            distractor=smart_index(opts.pe_distract, index, 1),
            no_support=smart_index(opts.pe_no_support, index, 0),
            unique_rest=smart_index(opts.pe_unique_rest, index, 0))
        exemplar_sampler = partial(
            samplers.get_exemplar_inds, allowed_inds=splits["exemplar"][opts.pe_exemplars[index]],
            match_query_and_distractors=opts.match_query_and_distractors)
        sampler = samplers.make_data_sampler(
            class_sampler, exemplar_sampler,
            fs_relabel=splits["relabeling"][smart_index(opts.pe_fs_relabel_scheme, index, "None")],
            noise_scale=smart_index(opts.pe_noise_scale, index, opts.noise_scale_train),
            assign_query_label_random=smart_index(opts.pe_assign_query_label_random, index, 0))
        evaluators[name] = sampler(seeds[index], features)
    return evaluators


def load_evaluator_file(path):
    """Read one saved evaluator file into {name: {'examples', 'labels'}}."""
    import h5py
    import jax.numpy as jnp
    with h5py.File(path, "r") as handle:
        return {name: {field: jnp.asarray(handle[name][field][:])
                       for field in ("examples", "labels")}
                for name in handle}


def score_checkpoint(run_folder, iteration, evaluator_file, opts=None):
    """Score one checkpoint on a saved evaluator, using the authors' evaluate().

    Returns {evaluator_name: {metric: mean}}. Accuracy is argmax over all five
    labels; loss is mean query cross-entropy in nats. Dropout is zero in this
    model, so the evaluation key does not change the result.
    """
    import jax
    import numpy as np
    import main as upstream_main
    import opto
    opts = opts or load_run_options(run_folder)
    checkpoint = load_checkpoint(run_folder, iteration, opts)
    forward = opto.make_fn_from_opts(opts)
    assert forward is opto.default_model_fwd_fn, "run unexpectedly used an intervention"
    evaluators = load_evaluator_file(evaluator_file)
    scored = upstream_main.evaluate(checkpoint["model"], forward, jax.random.PRNGKey(0),
                                    evaluators, opts.eval_bs)
    return {name: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            for name, metrics in scored.items()}


def evaluation_seeds(opts):
    """The (data, model) evaluation key pair, split exactly as main.py:297,348."""
    import jax
    return jax.random.split(as_key(opts.eval_seed), 2)


def training_seeds(opts):
    """The (data, model) training key pair, split exactly as main.py:296,347."""
    import jax
    return jax.random.split(as_key(opts.train_seed), 2)


def as_key(seed):
    """Accept either a plain integer seed or an already-built PRNG key."""
    import jax
    import numpy as np
    return seed if np.ndim(seed) > 0 else jax.random.PRNGKey(int(seed))
