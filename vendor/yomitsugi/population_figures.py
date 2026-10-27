"""Population figures from saved observations, with fixed replicate selection."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def _resolved(root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    if (root / path).exists():
        return root / path
    if path.exists():
        return path
    # A moved archive remains readable using its last path component.
    return root / "results" / "population" / path.name


def _series(frame, column):
    values = [json.loads(value) for value in frame[column].fillna("{}")]
    keys = sorted({key for value in values for key in value})
    return {key: np.array([value.get(key, 0) for value in values]) for key in keys}


def _save(fig, directory, stem):
    for extension in ("svg", "png"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem


def render_population(root: Path) -> list[str]:
    """Return figure stems; missing population stage yields no fabricated plot.

    Select mixed repeat 0 for every condition before reading outcomes. The
    second figure uses all independent runs, including extinctions and limits.
    """
    root = Path(root)
    data = root / "results" / "population"
    if not (data / "latest.json").exists():
        return []
    latest = json.loads((data / "latest.json").read_text(encoding="utf-8"))
    run_dir = _resolved(root, latest["run_directory"])
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))["settings"]
    runs = pd.read_csv(data / "runs.csv")
    summary = pd.read_csv(data / "summary.csv")
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    conditions = plan["conditions"]
    chosen = []
    for condition in conditions:
        path = run_dir / f"{condition['name']}__mixed__r000" / "trajectory.csv"
        if path.exists():
            chosen.append((condition, pd.read_csv(path)))
    outputs = []
    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.labelsize": 10, "axes.titlesize": 12,
             "legend.fontsize": 8, "svg.fonttype": "none"}
    with plt.rc_context(style):
        if chosen:
            columns = len(chosen)
            fig, axes = plt.subplots(5, columns, figsize=(7 * columns, 14.5), sharex="col", squeeze=False)
            all_readers = sorted({key for _, frame in chosen for key in _series(frame, "reader_counts")})
            reader_colors = {key: plt.get_cmap("tab20")(i % 20) for i, key in enumerate(all_readers)}
            template_colors = ["#087F8C", "#B83B76", "#BD7D00", "#4866A8"]
            for col, (condition, frame) in enumerate(chosen):
                x = frame["sweep"].to_numpy()
                axes[0, col].plot(x, frame["N"], color="#203A5F", linewidth=1.3)
                axes[0, col].set_title(f"{condition['name']}\nMixed population; preselected repeat 0")
                axes[0, col].set_ylabel("Total population")
                templates = _series(frame, "template_counts")
                for i, (key, values) in enumerate(templates.items()):
                    axes[1, col].plot(x, values, label=f"b:g = {key}", color=template_colors[i % len(template_colors)], linewidth=1.2)
                axes[1, col].set_ylabel("Information-word count")
                axes[1, col].legend(loc="upper right", frameon=False)
                readers = _series(frame, "reader_counts")
                keys = [key for key in all_readers if key in readers]
                axes[2, col].stackplot(x, [readers[key] for key in keys], colors=[reader_colors[key] for key in keys], linewidth=0)
                axes[2, col].set_ylabel("Actual reader counts\n(stacked p states)")
                axes[3, col].plot(x, frame["S"], color="#92631A", linewidth=1.2)
                axes[3, col].axhline(plan["Smax"], color="#BBBBBB", linestyle=":", linewidth=.8)
                axes[3, col].set_ylabel("Resource remaining")
                directions = _series(frame, "direction_births")
                for i, (key, values) in enumerate(directions.items()):
                    axes[4, col].plot(x, values, color=template_colors[i % len(template_colors)],
                                      linewidth=.8, alpha=.85, label=key)
                axes[4, col].set_ylabel("Births per sweep\n(actor origin -> target origin)")
                axes[4, col].set_xlabel("Sweep (finite observation)")
                axes[4, col].legend(loc="upper right", frameon=False, ncol=2)
                for row in range(5):
                    axes[row, col].set_ylim(bottom=0)
                    axes[row, col].set_xlim(0, plan["sweeps"])
                    axes[row, col].grid(axis="y", alpha=.18)
                status = str(frame["termination_reason"].iloc[0]) if len(frame) else "empty"
                if status != "normal_completion":
                    axes[0, col].text(.02, .93, f"Stopped: {status}", transform=axes[0, col].transAxes, color="#AA2222")
            handles = [Line2D([0], [0], color=reader_colors[key], linewidth=5,
                              label=f"p={key.replace(',', '')}") for key in all_readers]
            fig.legend(handles=handles, title="Actual reader words (same colors in every panel)",
                       loc="lower center", ncol=min(8, max(1, len(handles))), frameon=False,
                       bbox_to_anchor=(.5, .015), fontsize=8, title_fontsize=9)
            fig.suptitle("Population and reader dynamics are distinct observations", fontsize=17, y=.992)
            fig.text(.5, .005, "Labels record ancestry only. Display colors and positions do not enter reactions. Trajectories were not selected by outcome.",
                     ha="center", fontsize=9, color="#555555")
            fig.tight_layout(rect=(0, .09, 1, .975), h_pad=1.6)
            outputs.append(_save(fig, figures, "population_trajectories"))

        modes = plan["modes"]
        labels = {"single_a": "Single A", "single_b": "Single B", "mixed": "Mixed",
                  "pooled_shared": "Fixed pools\nshared resource", "pooled_separate": "Fixed pools\nseparate resource"}
        fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 6), squeeze=False)
        for col, condition in enumerate(conditions):
            ax = axes[0, col]
            condition_runs = runs[runs.condition == condition["name"]]
            group_sizes = []
            lower_bounds = []
            for position, mode in enumerate(modes):
                subset = condition_runs[condition_runs["mode"] == mode].sort_values("repeat_id")
                values = subset.N_final.to_numpy(dtype=float)
                group_sizes.append(len(values))
                # Deterministic spread by replicate ID, without plotting RNG.
                spread = np.linspace(-.14, .14, len(values)) if len(values) > 1 else np.zeros(1)
                completed = subset.termination_reason.to_numpy() == "normal_completion"
                ax.scatter(position + spread[completed], values[completed], color="#477DA8", alpha=.72, s=25, zorder=3)
                if not completed.all():
                    ax.scatter(position + spread[~completed], values[~completed], marker="x", color="#B83333", s=40, zorder=4)
                row = summary[(summary.condition == condition["name"]) & (summary["mode"] == mode) & (summary.measure == "N_final")]
                if len(row):
                    mean = float(row.iloc[0]["mean"])
                    low, high = row.iloc[0].mean_ci95_low, row.iloc[0].mean_ci95_high
                    if pd.notna(low) and pd.notna(high):
                        lower_bounds.append(float(low))
                        ax.errorbar(position, mean, yerr=[[mean - float(low)], [float(high) - mean]],
                                    color="#111111", marker="D", markersize=4, capsize=5, linewidth=1.4, zorder=5)
                    else:
                        ax.plot(position, mean, marker="D", color="#111111", markersize=4)
                extinct = int(subset.extinct.sum())
                ax.text(position, 1.015, f"extinct {extinct}/{len(values)}", ha="center", va="bottom", fontsize=8,
                        transform=ax.get_xaxis_transform())
            sample_sizes = str(group_sizes[0]) if len(set(group_sizes)) == 1 else ','.join(map(str,group_sizes))
            ax.set_title(f"{condition['name']}\n{plan['sweeps']} sweeps; n={sample_sizes} independent runs per group", pad=36)
            local_labels = dict(labels, single_a=f"Single A\n{condition['types'][0]['name']}",
                                single_b=f"Single B\n{condition['types'][1]['name']}")
            ax.set_xticks(range(len(modes)), [local_labels.get(mode, mode) for mode in modes])
            ax.tick_params(axis="x", labelsize=9)
            ax.set_ylabel("Population at last observed sweep")
            ax.set_ylim(bottom=min([-3] + [v-3 for v in lower_bounds]))
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Independent-run distributions, including extinction", fontsize=16, y=1.02)
        fig.text(.5, .015, "Blue: independent runs. Black: mean and Student-t 95% interval. Red x: stopped before planned end.\n"
                 "Fixed-pool controls impose contacts explicitly; compare shared vs separate resources within those controls.",
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .12, 1, 1))
        outputs.append(_save(fig, figures, "population_run_distributions"))
    return outputs
