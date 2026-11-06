"""Build the CAAI figures and tables from the released interval rows."""

import argparse
import csv
import gzip
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "figure_values.csv"
CAPACITY_DATA = ROOT / "data" / "capacity_values.csv"
INK, MUTED, GRID = "#172B42", "#4C5F72", "#D9E3EB"
BLUE, AMBER, GREEN = "#145C83", "#A54B1B", "#217366"
COHORTS = ("H", "H2")
ENDPOINTS = ("J_R", "J_S", "delta")
MODELS = (("qwen3_vl_4b", "Qwen3-VL-4B"), ("internvl35_8b", "InternVL3.5-8B"))
PROMPTS = (("direct_short", "direct"), ("task_constrained_short", "constrained"))
WEIGHTS = (0, .25, .50, .75, 1)
PANEL_ROWS = (
    ("seed/TVn/original_mix/ALL", r"Unscaled \tvn"),
    ("seed/block_TVn/original_mix/ALL", r"Scaled \tvn"),
    ("seed/PCA32_TVn/original_mix/ALL", r"PCA32 \tvn"),
    ("question/block_TQn/original_mix/ALL", r"Scaled \tqn"),
    ("question/PCA32_TQn/original_mix/ALL", r"PCA32 \tqn"),
)
NATIVE_CELL = "seed/TVn/native_only/ALL"
TREE_CELL = "strong/Tn/strong_mix/ALL"
VIDEO_ROWS = (
    (NATIVE_CELL, "Native only"),
    PANEL_ROWS[0],
    PANEL_ROWS[1],
    (TREE_CELL, r"Boosted trees, $\T+\Nfeat$"),
    PANEL_ROWS[3],
)
LEARNER_ROWS = (
    ("seed/TVn/ridge_original/ALL", r"Ridge (3), unscaled \tvn"),
    ("seed/block_TVn/ridge_original/ALL", r"Ridge (3), scaled \tvn"),
    ("seed/TVn/ridge_expanded/ALL", r"Ridge (6), unscaled \tvn"),
    ("seed/block_TVn/ridge_expanded/ALL", r"Ridge (6), scaled \tvn"),
    (TREE_CELL, r"Boosted trees, $\T+\Nfeat$"),
    ("strong/PCA32_TVn/strong_mix/ALL", r"Boosted trees, PCA32 \tvn"),
)
FIELDS = ("dataset", "cohort", "endpoint", "left", "right", "estimate", "low", "high")


class Effect(NamedTuple):
    estimate: float
    low: float
    high: float


def weight_cell(weight, candidate, panel="ridge_only"):
    return f"block_weight/weighted_TVn_{weight:.2f}/{panel}/{candidate}"


def required_rows():
    for cohort in COHORTS:
        for dataset in ("vqav2", "activitynetqa"):
            for cell, _ in PANEL_ROWS:
                for endpoint in ENDPOINTS:
                    yield dataset, cohort, endpoint, cell, "zero"
        raw, scaled, _, question, _ = [cell for cell, _ in PANEL_ROWS]
        for cell, _ in (*VIDEO_ROWS, *LEARNER_ROWS):
            for endpoint in ENDPOINTS:
                yield "activitynetqa", cohort, endpoint, cell, "zero"
        for endpoint in ENDPOINTS:
            yield "activitynetqa", cohort, endpoint, raw, scaled
            yield "activitynetqa", cohort, endpoint, scaled, question
            for baseline, comparator in (
                (NATIVE_CELL, scaled), (NATIVE_CELL, question),
                (TREE_CELL, question),
                ("seed/TVn/ridge_expanded/ALL", "seed/block_TVn/ridge_expanded/ALL"),
            ):
                yield "activitynetqa", cohort, endpoint, baseline, comparator
        for model, _ in MODELS:
            for prompt, _ in PROMPTS:
                candidate = f"{model}__{prompt}"
                for weight in WEIGHTS:
                    yield "activitynetqa", cohort, "delta", weight_cell(weight, candidate), "zero"
                for endpoint in ENDPOINTS:
                    yield ("activitynetqa", cohort, endpoint, weight_cell(0, candidate),
                           weight_cell(1, candidate))
                for penalty in ("alpha_0.1", "alpha_1", "alpha_10"):
                    yield ("activitynetqa", cohort, "delta", weight_cell(0, candidate, penalty),
                           weight_cell(1, candidate, penalty))


def export_values(analysis):
    wanted = list(dict.fromkeys(required_rows()))
    wanted_set = set(wanted)
    selected = {}
    for path, opener in (
        (analysis / "cell_intervals.csv", open),
        (analysis / "pairwise_intervals.csv.gz", gzip.open),
    ):
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = tuple(row[field] for field in FIELDS[:5])
                reverse = (*key[:3], key[4], key[3])
                if key not in wanted_set and reverse not in wanted_set:
                    continue
                effect = Effect(*(float(row[name]) for name in (
                    "estimate_per_1000",
                    "simultaneous95_low_per_1000",
                    "simultaneous95_high_per_1000",
                )))
                if key in wanted_set:
                    selected[key] = effect
                else:
                    selected[reverse] = Effect(-effect.estimate, -effect.high, -effect.low)
    missing = wanted_set - selected.keys()
    if missing:
        raise ValueError(f"Missing required interval rows: {sorted(missing)[:5]}")
    with DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows((*key, *selected[key]) for key in wanted)


def export_capacity(analysis):
    cells = {cell for cell, _ in VIDEO_ROWS}
    rows = []
    for cohort in COHORTS:
        path = analysis / "activitynetqa" / cohort / "capacity_points.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(row for row in csv.DictReader(stream) if row["cell_id"] in cells)
    fields = ("dataset", "cohort", "cell_id", "capacity", "selection_capacity",
              "incoming_records", "accepted_records", "J_R", "J_S", "delta", "role")
    with CAPACITY_DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_values():
    with DATA.open(encoding="utf-8", newline="") as stream:
        return {
            tuple(row[field] for field in FIELDS[:5]):
                Effect(*(float(row[field]) for field in FIELDS[5:]))
            for row in csv.DictReader(stream)
        }


def setup_plotting():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def draw_pipeline():
    """Show the actual grouped split, selection and queue operations."""
    with plt.rc_context({
        "font.family": "Arial",
        "font.size": 8,
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }):
        fig = plt.figure(figsize=(7.05, 4.16))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(-3, 507), ylim=(-3, 298))
        ax.axis("off")
        graphite = "#343D46"
        light = "#D4D9DE"
        pale_blue, pale_amber = "#DCE9F0", "#F1DFD4"

        def text(x, y, value, size=8, color=graphite, weight="normal", ha="left"):
            ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                    ha=ha, va="center")

        def rect(x, y, width, height, fill="white", stroke=graphite, lw=.65):
            ax.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                                   edgecolor=stroke, linewidth=lw))

        def arrow(points, color=graphite, dashed=False, lw=.7):
            if len(points) > 2:
                xs, ys = zip(*points[:-1])
                ax.plot(xs, ys, color=color, lw=lw, ls="--" if dashed else "-")
            ax.add_patch(FancyArrowPatch(
                points[-2], points[-1], arrowstyle="-|>", mutation_scale=6,
                shrinkA=0, shrinkB=0, linewidth=lw, color=color,
                linestyle="--" if dashed else "-",
            ))

        def queue(x, y, order, accepted, color, tint, width=11.4, gap=1.6):
            for index, record in enumerate(order):
                active = index < accepted
                rect(x + index * (width + gap), y, width, 13,
                     fill=tint if active else "white",
                     stroke=color if active else "#AAB2B9", lw=.7)
                text(x + index * (width + gap) + width / 2, y + 6.5,
                     str(record), size=7.4,
                     color=graphite if active else "#7A838C", ha="center")

        # An equivalent encoding is applied before either development flow.
        text(0, 287, "Equivalent visual encodings", size=8.4, weight="bold")
        text(186, 287, r"$V$", size=10.5, ha="center")
        text(253, 287, r"$[V,V,V,V]$", size=10.5, ha="center")
        text(326, 287, r"$VO$", size=10.5, ha="center")
        text(502, 290, "Same answers and sources", size=7.2, ha="right")
        text(502, 279, r"$OO^\top=I$", size=8.2, ha="right")
        ax.plot([0, 502], [270, 270], color=light, lw=.7)

        # The matrices encode membership by fill, keeping source identity by column.
        text(0, 253, "(a) Development splits", size=8.7, weight="bold")
        text(0, 237, r"Source columns from the same $D$", size=7.5)
        rect(1, 219, 6.8, 6.8, fill=graphite)
        text(12, 222.4, r"fit $\mathcal{F}$", size=7.5)
        rect(61, 219, 6.8, 6.8)
        text(72, 222.4, r"validate $\mathcal{V}$", size=7.5)
        flow_specs = (
            (187, "R", "Record-random", BLUE, pale_blue),
            (130, "S", "Source-disjoint", AMBER, pale_amber),
        )
        for center, flow, name, color, _ in flow_specs:
            text(0, center + 20, f"{flow}  {name}", size=8.2, color=color, weight="bold")
            for source in range(10):
                for question in range(3):
                    if flow == "R":
                        fit_record = source == 9 or question != source % 3
                    else:
                        fit_record = source < 7
                    rect(3 + source * 10.4, center - 10.8 + question * 7.9,
                         6.8, 6.8, fill=graphite if fit_record else "white", lw=.45)
            relation = r"\ne\varnothing" if flow == "R" else r"=\varnothing"
            text(52, center - 21, r"$\mathcal{G}_{\mathcal{F}}\cap"
                 r"\mathcal{G}_{\mathcal{V}}" + relation + "$",
                 size=8.8, ha="center")
            arrow([(111, center), (144, center)], color=color)

        # The upper ribbon denotes the feature design, common to the two flows.
        text(145, 253, "(b) Fit and select a policy", size=8.7, weight="bold")
        text(145, 237, "Frozen model and prompt candidates", size=7.5)
        text(148, 220, r"$z$", size=9)
        for x, width, name in ((160, 20, r"$T$"), (180, 63, r"$V\ /\ Q$"), (243, 20, r"$n$")):
            rect(x, 213, width, 14, fill="#F1F3F5", stroke="#8B959E", lw=.55)
            text(x + width / 2, 220, name, size=8.7, ha="center")

        for center, flow, _, color, _ in flow_specs:
            rect(146, center - 11, 48, 22, lw=.7)
            text(170, center, r"$\widehat r_{" + flow + r",z}$", size=11, ha="center")
            text(170, center - 21, r"fit on $\mathcal{F}_{" + flow + "}$",
                 size=7.5, ha="center")
            arrow([(195, center), (216, center)])
            rect(217, center - 11, 55, 22, lw=.7)
            text(244.5, center, r"$\arg\min\,J_{\mathcal{V}}$", size=10.2, ha="center")
            text(244.5, center - 21, r"select on $\mathcal{V}_{" + flow + "}$",
                 size=7.5, ha="center")
            arrow([(140, center), (140, center + 18),
                   (244.5, center + 18), (244.5, center + 12)],
                  color=graphite, dashed=True, lw=.65)
            text(204, center + 22, "native score", size=7.1, ha="center")
            arrow([(274, center), (324, center)], color=color, lw=.8)
            text(299, center + 9.5, r"$\widehat p_{" + flow + "}$",
                 size=10.2, color=color, ha="center")
        text(145, 91, "Fit normalization on fitting records only", size=7.1)

        # Every queue uses the same ten illustrative records and the same capacity.
        text(326, 253, "(c) Fixed-capacity evaluation", size=8.7, weight="bold")
        text(477, 237, r"$H/H_2$", size=8.5, ha="center")
        source_groups = (("a", (1, 2)), ("b", (3, 4, 5)), ("c", (6, 7, 8, 9, 10)))
        bank_x, pitch, cell_width = 326, 13, 11.4
        offset = 0
        for source, records in source_groups:
            left = bank_x + offset * pitch
            right = left + len(records) * pitch - 1.6
            ax.plot([left, left, right, right], [230, 232, 232, 230],
                    color="#7F8891", lw=.55)
            text((left + right) / 2, 236, source, size=7.8, ha="center")
            for record in records:
                rect(bank_x + offset * pitch, 213, cell_width, 13,
                     fill="#F1F3F5", stroke="#AAB2B9", lw=.5)
                text(bank_x + offset * pitch + cell_width / 2, 219.5,
                     str(record), size=7.4, ha="center")
                offset += 1
        text(477, 219.5, r"$N=10$", size=8.1, ha="center")
        orders = {
            "R": (7, 1, 4, 8, 2, 6, 3, 10, 5, 9),
            "S": (2, 8, 6, 3, 10, 1, 5, 4, 7, 9),
        }
        cutoff = bank_x + 7 * pitch - .8
        ax.plot([cutoff, cutoff], [115, 202], color="#5E6872", lw=.7, ls=(0, (2.5, 2)))
        text(cutoff, 205, r"$K$", size=9, ha="center")
        for center, flow, _, color, tint in flow_specs:
            queue(bank_x, center - 6.5, orders[flow], 7, color, tint)
            text(477, center, r"$J_{" + flow + "}$", size=10.8, color=color, ha="center")
        text(370, 111, "accepted", size=7.4, ha="center")
        text(438, 111, "deferred", size=7.4, ha="center", color="#7A838C")
        text(410, 94, r"$J_g=N^{-1}\sum_{i\in A_g}L_i,\quad q=0.70$",
             size=9.4, ha="center")

        # One illustrative source draw is applied to both complete ranked lists.
        ax.plot([0, 502], [78, 78], color=light, lw=.7)
        text(0, 65, "(d) Paired source resampling", size=8.7, weight="bold")
        text(274, 65, "Same draw across policies and development seeds", size=7.5)
        for index, source in enumerate(("a", "a", "c")):
            rect(index * 21, 28, 17, 20, stroke="#7F8891")
            text(index * 21 + 8.5, 38, source, size=9, ha="center")
        text(29.5, 16, r"$m^*=(2,0,1)$", size=8.5, ha="center")
        arrow([(64, 38), (103, 38)])
        ax.plot([103, 103], [26, 47], color=graphite, lw=.7)
        source_of = {record: source for source, records in source_groups for record in records}
        multiplicities = {"a": 2, "b": 0, "c": 1}
        for center, flow, color, tint in ((47, "R", BLUE, pale_blue), (26, "S", AMBER, pale_amber)):
            expanded = [record for record in orders[flow]
                        for _ in range(multiplicities[source_of[record]])]
            arrow([(103, center), (124, center)], color=color)
            text(118, center + 8, flow, size=7.3, color=color, ha="center")
            queue(127, center - 6.5, expanded, 6, color, tint, width=10.4, gap=1.4)
        resampled_cutoff = 127 + 6 * 11.8 - .7
        ax.plot([resampled_cutoff, resampled_cutoff], [17, 56],
                color="#5E6872", lw=.7, ls=(0, (2.5, 2)))
        text(179, 7, r"$N^*=9,\quad K^*=\lfloor qN^*\rfloor=6$", size=8.4, ha="center")
        ax.plot([240, 247, 247, 240], [47, 47, 26, 26], color=graphite, lw=.7)
        arrow([(247, 36.5), (266, 36.5)])
        text(326, 36.5, r"$\Delta^*=\overline{J}_R^*-\overline{J}_S^*$",
             size=11, ha="center")
        arrow([(388, 36.5), (410, 36.5)])
        text(456, 42, "Joint intervals", size=8.4, ha="center", weight="bold")
        text(456, 27, r"$\{J_R,J_S,\Delta\}$", size=9.4, ha="center")

        for extension in ("pdf", "svg", "png"):
            fig.savefig(ROOT / "figures" / f"pipeline.{extension}", dpi=300)
        plt.close(fig)


def draw_effects(values):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.20), sharex=True, sharey=True)
    fig.subplots_adjust(left=.211, right=.99, bottom=.265, top=.865, wspace=.18)
    unscaled, scaled = [cell for cell, _ in PANEL_ROWS[:2]]
    labels = (r"Record-random $J_\mathrm{R}$", r"Source-disjoint $J_\mathrm{S}$",
              r"Split contrast $\Delta$")
    for index, (axis, cohort) in enumerate(zip(axes, COHORTS)):
        name = "$H$" if cohort == "H" else "$H_2$"
        axis.set_title(f"({chr(97 + index)})  ActivityNet-QA {name}", loc="left",
                       fontsize=9.1, fontweight="bold")
        axis.set(xlim=(-.6, 18), ylim=(-.45, 2.62), xticks=(0, 5, 10, 15))
        axis.set_yticks((2, 1, 0), labels)
        axis.tick_params(axis="y", length=0, pad=6)
        axis.tick_params(axis="x", length=3)
        axis.spines[["top", "left", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color=GRID, linewidth=.65)
        axis.axvline(0, color=MUTED, lw=.8, ls="--")
        for y, endpoint, color, marker in zip(
            (2, 1, 0), ENDPOINTS, (BLUE, AMBER, GREEN), ("o", "s", "D"),
        ):
            e = values["activitynetqa", cohort, endpoint, unscaled, scaled]
            axis.errorbar(e.estimate, y, xerr=[[e.estimate - e.low], [e.high - e.estimate]],
                          fmt=marker, color=color, markersize=4.9, lw=1.6, capsize=3)
            axis.annotate(f"{e.estimate:.2f}", (e.estimate, y), xytext=(0, 6),
                          textcoords="offset points", ha="center", va="bottom",
                          fontsize=8.0, color=color, fontweight="bold")
    fig.supxlabel("Reduction after scaling (errors per 1,000 arrivals)", x=.603, y=.036,
                  fontsize=8.5)
    fig.savefig(ROOT / "figures" / "scaling_effects.pdf")
    plt.close(fig)


def draw_paths(values):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.72), sharex=True, sharey=True)
    fig.subplots_adjust(left=.107, right=.985, bottom=.30, top=.87, wspace=.16)
    for index, (axis, cohort) in enumerate(zip(axes, COHORTS)):
        name = "$H$" if cohort == "H" else "$H_2$"
        axis.set_title(f"({chr(97 + index)})  ActivityNet-QA {name}", loc="left",
                       fontsize=9.1, fontweight="bold")
        axis.set(xticks=WEIGHTS, xticklabels=("0", ".25", ".50", ".75", "1"),
                 xlim=(-.03, 1.03), ylim=(-.9, 9.3), yticks=(0, 2, 4, 6, 8))
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=3)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=GRID, linewidth=.65)
        axis.axhline(0, color=MUTED, lw=.85, zorder=1)
        for (model, model_label), color, marker in zip(MODELS, (BLUE, AMBER), ("o", "s")):
            for (prompt, prompt_label), line in zip(PROMPTS, ("-", "--")):
                candidate = f"{model}__{prompt}"
                points = [values["activitynetqa", cohort, "delta",
                                 weight_cell(weight, candidate), "zero"].estimate
                          for weight in WEIGHTS]
                axis.plot(WEIGHTS, points, color=color, marker=marker, linestyle=line,
                          markerfacecolor=color if line == "-" else "white",
                          linewidth=1.25, markersize=3.7, label=f"{model_label}, {prompt_label}")
    fig.supylabel(r"$\Delta$ (errors per 1,000 arrivals)", x=.013, y=.585, fontsize=8.5)
    fig.supxlabel(r"Block-weight exponent $\lambda$", x=.546, y=.17, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54, .002),
               ncol=2, frameon=False, fontsize=7.9, columnspacing=1.55)
    fig.savefig(ROOT / "figures" / "weight_paths.pdf")
    plt.close(fig)


def draw_capacity(values):
    with CAPACITY_DATA.open(encoding="utf-8", newline="") as stream:
        rows = {(row["cohort"], row["cell_id"], float(row["capacity"])): row
                for row in csv.DictReader(stream)}
    capacities = (.1, .3, .5, .7, .9)
    for cohort in COHORTS:
        for cell, _ in VIDEO_ROWS:
            for endpoint in ENDPOINTS:
                observed = float(rows[cohort, cell, .7][endpoint]) * 1000
                expected = values["activitynetqa", cohort, endpoint, cell, "zero"].estimate
                if abs(observed - expected) > 1e-8:
                    raise ValueError(f"Capacity/table mismatch: {cohort}, {cell}, {endpoint}")
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 3.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.985, bottom=.24, top=.92, wspace=.13, hspace=.35)
    styles = (
        (PANEL_ROWS[0][0], r"Unscaled $T+V+n$", "#727B83", "--", "v"),
        (PANEL_ROWS[1][0], r"Scaled $T+V+n$", BLUE, "-", "o"),
        (TREE_CELL, r"Boosted trees, $T+n$", AMBER, "--", "s"),
        (PANEL_ROWS[3][0], r"Scaled $T+Q+n$", GREEN, "-", "D"),
    )
    for row_index, cohort in enumerate(COHORTS):
        for column_index, endpoint in enumerate(("J_R", "J_S")):
            axis = axes[row_index, column_index]
            letter = chr(97 + row_index * 2 + column_index)
            cohort_label = "$H$" if cohort == "H" else "$H_2$"
            flow_label = "Record-random" if column_index == 0 else "Source-disjoint"
            axis.set_title(f"({letter})  {cohort_label}: {flow_label}", loc="left",
                           fontsize=8.8, fontweight="bold", pad=5)
            axis.set(xticks=capacities, xticklabels=(".1", ".3", ".5", ".7", ".9"),
                     xlim=(.07, .93), ylim=(-23, 25), yticks=(-20, -10, 0, 10, 20))
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(length=3)
            axis.grid(axis="y", color=GRID, linewidth=.5)
            axis.axhline(0, color=MUTED, lw=.85)
            for cell, label, color, line, marker in styles:
                gains = [1000 * (float(rows[cohort, NATIVE_CELL, q][endpoint])
                                  - float(rows[cohort, cell, q][endpoint]))
                         for q in capacities]
                axis.plot(capacities, gains, label=label, color=color, linestyle=line,
                          marker=marker, lw=1.1, markersize=3.3,
                          markerfacecolor="white" if line == "--" else color)
    fig.supylabel("Reduction from native score\n(errors per 1,000 arrivals)",
                  x=.008, y=.60, fontsize=8.4)
    fig.supxlabel(r"Acceptance fraction $q$", x=.54, y=.13, fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54, -.005),
               ncol=2, frameon=False, fontsize=8, columnspacing=2)
    fig.savefig(ROOT / "figures" / "capacity_sensitivity.pdf")
    plt.close(fig)


def tex_value(number):
    return f"{0.0 if abs(number) < .005 else number:.2f}"


def write_policy_rows(values, name, datasets, selected):
    lines = ["% Generated by build_figures.py from data/figure_values.csv."]
    for dataset in datasets:
        for cohort in COHORTS:
            if len(lines) > 1:
                lines.append(r"\addlinespace[3pt]")
            cohort_label = "$H$" if cohort == "H" else "$H_2$"
            if len(datasets) > 1:
                cohort_label = ("VQA v2" if dataset == "vqav2" else "Video") + " " + cohort_label
            for index, (cell, representation) in enumerate(selected):
                r, s, d = [values[dataset, cohort, endpoint, cell, "zero"] for endpoint in ENDPOINTS]
                fields = (cohort_label if index == 0 else "", representation,
                          tex_value(r.estimate), tex_value(s.estimate),
                          "$" + tex_value(d.estimate) + "$",
                          f"$[{tex_value(d.low)},{tex_value(d.high)}]$")
                lines.append(" & ".join(fields) + r" \\")
    (ROOT / "sections" / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_penalty_rows(values):
    lines = ["% Generated by build_figures.py from data/figure_values.csv."]
    for cohort in COHORTS:
        label = "$H$" if cohort == "H" else "$H_2$"
        for index, (panel, penalty) in enumerate((("alpha_0.1", "0.1"), ("alpha_1", "1"), ("alpha_10", "10"))):
            contrasts = []
            for model, _ in MODELS:
                for prompt, _ in PROMPTS:
                    candidate = f"{model}__{prompt}"
                    contrasts.append(values["activitynetqa", cohort, "delta",
                                            weight_cell(0, candidate, panel),
                                            weight_cell(1, candidate, panel)])
            estimates = [e.estimate for e in contrasts]
            range_text = f"{tex_value(min(estimates))}--{tex_value(max(estimates))}"
            bound = tex_value(min(e.low for e in contrasts))
            lines.append(f"{label if index == 0 else ''} & {penalty} & {range_text} & {bound}" + r" \\")
        if cohort == "H":
            lines.append(r"\addlinespace[3pt]")
    (ROOT / "sections" / "fixed_penalty_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, help="Refresh data from the full released CSV files")
    args = parser.parse_args()
    if args.analysis_dir is not None:
        export_values(args.analysis_dir)
        export_capacity(args.analysis_dir)
    values = load_values()
    setup_plotting()
    draw_pipeline()
    draw_effects(values)
    draw_paths(values)
    draw_capacity(values)
    write_policy_rows(values, "video_policy_rows.tex", ("activitynetqa",), VIDEO_ROWS)
    write_policy_rows(values, "learner_rows.tex", ("activitynetqa",), LEARNER_ROWS)
    write_policy_rows(values, "mixed_panel_rows.tex", ("vqav2", "activitynetqa"), PANEL_ROWS)
    write_penalty_rows(values)


if __name__ == "__main__":
    main()
