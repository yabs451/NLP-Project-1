"""Record the supplied HDF5 data, baseline defaults and real sampler examples."""
from functools import partial
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.dont_write_bytecode = True
from run_baseline import ROOT, UPSTREAM, baseline_arguments
sys.path.insert(0, str(UPSTREAM))

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import main_utils
import opto
import samplers


def main():
    output = ROOT / "results" / "inspection"
    output.mkdir(parents=True, exist_ok=True)
    parser = main_utils.create_parser()
    opto.add_args_to_parser(parser)
    opts = parser.parse_args(baseline_arguments())
    opts.train_microbs = opts.train_bs
    opts.model_output_classes = opts.fs_relabel
    main_utils.check_opts(opts)
    (output / "baseline_options.json").write_text(json.dumps(vars(opts), indent=2))
    filename = Path(opts.data_file)
    structure = {}
    with h5py.File(filename, "r") as handle:
        def inspect(name, obj):
            structure[name] = {"kind": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                               "attributes": {k: str(v) for k, v in obj.attrs.items()}}
            if isinstance(obj, h5py.Dataset):
                structure[name].update(shape=list(obj.shape), dtype=str(obj.dtype),
                                       compression=obj.compression)
        handle.visititems(inspect)
        features = handle[opts.data_path_in_file][:]
        class_order = handle["class_order"][:]
    np.testing.assert_array_equal(np.sort(class_order), np.arange(features.shape[0]))
    assert np.isfinite(features).all(), "Non-finite input features"
    splits = main_utils.get_splits_from_opts(opts, features.shape)
    class_sampler = partial(samplers.get_constant_burst_seq_idxs,
        classes=splits["class"]["train"], class_distr=jnp.ones(50) / 50,
        num_seqs=32, context_len=2, burstiness=1)
    mixed = partial(samplers.get_mixed_seq_idxs, mix_probabilities=jnp.array([1.0]),
                    mix_substrate_fns=[class_sampler])
    exemplars = partial(samplers.get_exemplar_inds, allowed_inds=splits["exemplar"]["train"],
                        match_query_and_distractors=False)
    sampler = samplers.make_data_sampler(mixed, exemplars, fs_relabel=splits["relabeling"]["train"])
    # Same key splitting as main.py: these are the actual first training batch.
    data_seed, _ = jax.random.split(jax.random.PRNGKey(opts.train_seed))
    _, batch_seed = jax.random.split(data_seed)
    class_key, exemplar_key, label_key = jax.random.split(batch_seed, 3)
    classes = mixed(class_key)
    exemplar_ids = np.asarray(exemplars(exemplar_key, classes["idx_types"]))
    class_ids = np.asarray(classes["class_idxs"])
    batch = sampler(batch_seed, jnp.asarray(features))
    labels = np.asarray(batch["labels"])
    np.testing.assert_array_equal(batch["examples"], features[class_ids, exemplar_ids])
    support = class_ids[:, :-1] == class_ids[:, -1:]
    assert np.all(support.sum(axis=1) == 1)
    assert np.all(labels[:, :-1][support] == labels[:, -1])
    assert np.all(labels[:, 0] != labels[:, 1])
    assert set(map(tuple, np.sort(labels[:, :2], axis=1))) <= set(map(tuple, np.asarray(splits["relabeling"]["train"])))
    # Hold classes fixed; observe the random per-sequence reassignment directly.
    relabeled = np.asarray(samplers.fewshot_relabel(jax.random.PRNGKey(123),
        **classes, labels=splits["relabeling"]["train"]))
    assert np.any(labels != relabeled)
    examples = [{"classes": c.tolist(), "exemplars": e.tolist(), "labels": y.tolist(),
                 "same_classes_new_labels": z.tolist()}
                for c, e, y, z in zip(class_ids[:5], exemplar_ids[:5], labels[:5], relabeled[:5])]
    git = ["git", "-c", "safe.directory=" + UPSTREAM.as_posix(), "-C", str(UPSTREAM)]
    metadata = {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "windows": list(platform.win32_ver()),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "jax_devices": [str(d) for d in jax.devices()], "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "upstream_commit": subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip(),
        "upstream_status": subprocess.check_output(git + ["status", "--short"], text=True).strip(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
        "data": {"path": str(filename), "bytes": filename.stat().st_size,
                 "sha256": hashlib.sha256(filename.read_bytes()).hexdigest(), "structure": structure,
                 "class_order_is_permutation": True, "class_order_first_10": class_order[:10].tolist(),
                 "feature_min": float(features.min()), "feature_max": float(features.max()), "finite": True},
        "label_pairs": {name: np.asarray(splits["relabeling"][name]).tolist() for name in ("train", "val", "test")},
        "sample_shapes": {name: list(value.shape) for name, value in batch.items()},
        "sample_dtypes": {name: str(value.dtype) for name, value in batch.items()},
        "examples": examples, "data_checks_passed": True}
    (output / "inspection.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
