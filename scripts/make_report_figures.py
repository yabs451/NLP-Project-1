"""Build report figures from saved comparison JSON only.

This script does not import project model code, load checkpoints, evaluate data,
or modify any experiment result. It writes presentation-only figures beneath
``results/report_figures``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "report_figures"

SERIES = (
    (
        "Base",
        REPO_ROOT / "results" / "base_task" / "recursive"
        / "generation_comparison.json",
        "dev_accuracy",
        "#222222",
        "o",
        "-",
    ),
    (
        "Label argmax",
        REPO_ROOT / "results" / "extended_task" / "recursive" / "experiments"
        / "label_argmax" / "generation_comparison.json",
        "query_label_accuracy",
        "#0072B2",
        "s",
        "-",
    ),
    (
        "Label sampling T=3",
        REPO_ROOT / "results" / "extended_task" / "recursive" / "experiments"
        / "label_sampling_temperature_3" / "generation_comparison.json",
        "query_label_accuracy",
        "#D55E00",
        "^",
        "-",
    ),
    (
        "Label sampling T=5",
        REPO_ROOT / "results" / "extended_task" / "recursive" / "experiments"
        / "label_sampling_temperature_5" / "generation_comparison.json",
        "query_label_accuracy",
        "#CC79A7",
        "v",
        "-",
    ),
    (
        "Context feedback T=1/3",
        REPO_ROOT / "results" / "extended_task" / "recursive" / "experiments"
        / "context_feedback_temperature_one_third" / "generation_comparison.json",
        "query_label_accuracy",
        "#009E73",
        "D",
        "--",
    ),
    (
        "Next-symbol T=0.2",
        REPO_ROOT / "results" / "extended_task" / "recursive" / "experiments"
        / "symbol_sampling_temperature_0.2" / "generation_comparison.json",
        "query_label_accuracy",
        "#E69F00",
        "P",
        "--",
    ),
)


def load_series(path: Path, accuracy_key: str) -> tuple[list[int], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved comparison file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("generations")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No generation rows in {path}")
    try:
        generations = [int(row["generation"]) for row in rows]
        accuracies = [float(row[accuracy_key]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected comparison schema in {path}") from exc
    return generations, accuracies


def make_query_accuracy_figure() -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for label, source, key, colour, marker, linestyle in SERIES:
        generations, accuracies = load_series(source, key)
        ax.plot(
            generations,
            accuracies,
            label=label,
            color=colour,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=5.5,
        )

    ax.set_xlabel("Recursive generation")
    ax.set_ylabel("Development query accuracy")
    ax.set_xticks(range(7))
    ax.set_ylim(0.15, 1.02)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", ncols=2, frameon=False, fontsize=8.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "query_accuracy_by_generation.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    output = make_query_accuracy_figure()
    print(f"Wrote {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
