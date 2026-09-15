"""Launch only the first, unmodified induction-head paper experiment."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "icl-dynamics"


def baseline_arguments():
    source = (UPSTREAM / "ih_paper_runs.sh").read_text()
    line = next(line for line in source.splitlines() if line.startswith("python main.py "))
    for name, value in {"MAIN_RUN_ITERS": "1000000", "INIT_SEED": "5",
                        "SAVE_FOLDER": "./ih_paper_reprod/main_paper_is5_ih3_pt2"}.items():
        line = line.replace("$" + name, value)
    args = shlex.split(line)[2:]
    args[args.index("--data_file") + 1] = str(UPSTREAM / "omniglot_resnet18_randomized_order_s0.h5")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the original million-sequence schedule")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-name", default=None)
    options = parser.parse_args()
    name = options.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_") + ("baseline_full" if options.full else "baseline_trial")
    if Path(name).name != name or name in (".", ".."):
        parser.error("run-name must be a single folder name")
    folder = ROOT / "results" / name
    args = baseline_arguments() + ["--base_folder", str(ROOT / "results"), "--run", name]
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
              "return_code": code, "full": options.full, "events": events}
    (folder / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
