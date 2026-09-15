"""Train generation 0: the authors' unmodified baseline, run as a subprocess.

This launches upstream/icl-dynamics/main.py with the authors' own baseline
arguments. We change only the output location and which evaluators run; every
scientific setting stays as published.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, UPSTREAM, DEV_EVALUATOR_FILE, baseline_arguments

RESULTS = ROOT / "results" / "base_task"


def replace_list_option(args, flag, values):
    """Replace the values following `flag` (up to the next --option) in place."""
    start = args.index(flag) + 1
    end = start
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    args[start:end] = values


def assignment_arguments(args):
    """Swap the authors' evaluator set for our assignment development protocol.

    Drops only `fsl_test_class`, so the reserved held-out classes are never
    scored, and loads our fixed 100-class development evaluator from file.
    Training settings, seeds and the model are untouched, so the training
    random-number stream is identical to reproduction mode.
    """
    if not DEV_EVALUATOR_FILE.exists():
        raise SystemExit("Missing {}. Run scripts/prepare_evaluation_data.py first."
                         .format(DEV_EVALUATOR_FILE))
    keep = [i for i, name in enumerate(["fsl_train", "fsl_val_rl", "fsl_train_valex", "fsl_test_class"])
            if name != "fsl_test_class"]
    for flag, original in (("--pe_names", ["fsl_train", "fsl_val_rl", "fsl_train_valex", "fsl_test_class"]),
                           ("--pe_classes", ["train", "train", "train", "test"]),
                           ("--pe_exemplars", ["train", "train", "val", "train"]),
                           ("--pe_fs_relabel_scheme", ["train", "val", "train", "train"]),
                           ("--pe_burstiness", ["1", "1", "1", "1"])):
        assert args[args.index(flag) + 1:args.index(flag) + 5] == original, "upstream evaluator list changed"
        replace_list_option(args, flag, [original[i] for i in keep])
    return args + ["--load_eval_data", str(DEV_EVALUATOR_FILE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the original million-sequence schedule")
    parser.add_argument("--protocol", choices=["reproduction", "assignment"], default="reproduction",
                        help="reproduction: the authors' four evaluators. "
                             "assignment: our held-out development classes, no final-test scoring.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-name", default=None)
    options = parser.parse_args()
    name = options.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_") + (
        "generation_0_full" if options.full else "generation_0_short")
    if Path(name).name != name or name in (".", ".."):
        parser.error("run-name must be a single folder name")
    folder = RESULTS / name
    args = baseline_arguments()
    if options.protocol == "assignment":
        args = assignment_arguments(args)
    args += ["--base_folder", str(RESULTS), "--run", name]
    if not options.full:
        # Keep batch size and all task/model settings; change schedules only.
        args += ["--train_iters", "3200", "--eval_every", "1600", "--ckpt_every", "3200",
                 "--save_eval_data", "eval_data.h5"]
    command = [sys.executable, "-u", str(UPSTREAM / "main.py"), *args]
    print(subprocess.list2cmdline(command), flush=True)
    if options.dry_run:
        return
    folder.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(JAX_PLATFORMS="cpu", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    (folder / "command.json").write_text(json.dumps({"argv": command, "cwd": str(ROOT),
        "environment": {k: environment[k] for k in ("JAX_PLATFORMS", "WANDB_MODE", "PYTHONDONTWRITEBYTECODE")}}, indent=2))
    start = perf_counter()
    events = []
    with (folder / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            if line.startswith("----------") or line.startswith("End of training"):
                events.append({"event": line.strip(), "elapsed_seconds": perf_counter() - start})
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    timing = {"wall_seconds_including_imports_compilation_io": perf_counter() - start,
              "return_code": code, "full": options.full, "protocol": options.protocol, "events": events}
    (folder / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
