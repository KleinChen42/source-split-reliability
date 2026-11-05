"""Plot the four fixed-candidate ridge-grid paths from the released intervals."""
from pathlib import Path
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extension_layout import WEIGHTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("intervals", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.intervals.open(newline="", encoding="utf-8") as stream:
        rows = {
            (row["cohort"], row["left"]): row
            for row in csv.DictReader(stream)
            if row["dataset"] == "activitynetqa"
            and row["endpoint"] == "delta"
            and row["left"].startswith("block_weight/")
        }

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(6.3, 3.9), sharex=True, sharey=True)
    models = [("qwen3_vl_4b", "Qwen3-VL-4B"), ("internvl35_8b", "InternVL3.5-8B")]
    prompts = [("direct_short", "Direct"), ("task_constrained_short", "Task-constrained")]
    for model_index, (model_id, model_name) in enumerate(models):
        for prompt_index, (prompt_id, prompt_name) in enumerate(prompts):
            axis = axes[model_index, prompt_index]
            axis.axhline(0, linewidth=0.7, color="#696969", zorder=1)
            for cohort, color, marker, line in [
                ("H", "#00669a", "o", "-"),
                ("H2", "#a43e20", "s", "--"),
            ]:
                candidate = f"{model_id}__{prompt_id}"
                path = [
                    rows[cohort, f"block_weight/weighted_TVn_{weight:.2f}/ridge_only/{candidate}"]
                    for weight in WEIGHTS
                ]
                means = [float(row["estimate_per_1000"]) for row in path]
                lows = [float(row["simultaneous95_low_per_1000"]) for row in path]
                highs = [float(row["simultaneous95_high_per_1000"]) for row in path]
                axis.plot(WEIGHTS, means, marker=marker, markersize=3.4,
                          linestyle=line, linewidth=1.15, color=color, label=cohort)
                axis.fill_between(WEIGHTS, lows, highs, color=color, alpha=0.13, linewidth=0)
            axis.set_title(f"{model_name}, {prompt_name}", fontsize=9)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xticks(WEIGHTS, ["0", ".25", ".50", ".75", "1"])
            axis.tick_params(length=3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.supxlabel(r"Block-weight exponent $\lambda$", y=0.015)
    figure.supylabel(r"$J_R-J_S$ (accepted errors per 1,000)", x=0.01)
    figure.tight_layout(rect=(0.03, 0.04, 1, 0.93), pad=0.85)
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
