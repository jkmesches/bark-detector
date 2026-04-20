#!/usr/bin/env python3
"""
Read bark_report.csv and emit four standalone PNGs for inclusion in the LaTeX report.

Outputs (into ./report/):
  fig_timeline.png    — hourly timeline (events vs. estimated barks)
  fig_hour_hist.png   — totals by hour of day
  fig_date_hour.png   — date x hour heatmap with ordinance-violation outlines
  fig_dow_hour.png    — day-of-week x hour heatmap
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

PROJECT = Path(__file__).resolve().parent
CSV_IN = PROJECT / "bark_report.csv"
OUT_DIR = PROJECT / "report"
OUT_DIR.mkdir(exist_ok=True)

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_rows():
    rows = []
    with CSV_IN.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "date": r["date"],
                "dow": r["dow"],
                "hour": int(r["hour"]),
                "n_events": int(r["n_events"]),
                "n_bark_clips": int(r["n_bark_clips"]),
                "n_barks_est": int(r["n_barks_est"]),
                "mean_score": float(r["mean_score"]),
                "violation": r.get("violation", "") or "",
            })
    return rows


def main():
    rows = load_rows()
    dates = sorted({r["date"] for r in rows})
    date_idx = {d: i for i, d in enumerate(dates)}

    by_hour_events = np.zeros(24, dtype=np.int64)
    by_hour_barks = np.zeros(24, dtype=np.int64)
    dow_hour = np.zeros((7, 24), dtype=np.int64)
    heat = np.zeros((len(dates), 24), dtype=np.int64)
    violation_cells: set[tuple[int, int]] = set()
    flagged_dates: set[str] = set()

    for r in rows:
        h = r["hour"]
        by_hour_events[h] += r["n_events"]
        by_hour_barks[h] += r["n_barks_est"]
        dow = DOW_NAMES.index(r["dow"])
        dow_hour[dow, h] += r["n_barks_est"]
        heat[date_idx[r["date"]], h] = r["n_barks_est"]
        if r["violation"]:
            violation_cells.add((date_idx[r["date"]], h))
            flagged_dates.add(r["date"])

    total_barks = int(sum(r["n_barks_est"] for r in rows))
    total_events = int(sum(r["n_events"] for r in rows))
    total_clips = int(sum(r["n_bark_clips"] for r in rows))

    # ---- Figure 1: hourly timeline ----
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    hourly_dt = [datetime.fromisoformat(r["date"]).replace(hour=r["hour"]) for r in rows]
    ev = [r["n_events"] for r in rows]
    bk = [r["n_barks_est"] for r in rows]
    ax.bar(hourly_dt, ev, width=1/24, color="#bfbfbf", label="detector events", align="edge")
    ax.bar(hourly_dt, bk, width=1/24, color="#b11d1d", label="est. barks", align="edge")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("count / hour")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_timeline.png", dpi=160)
    plt.close(fig)

    # ---- Figure 2: hour-of-day histogram ----
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(24)
    ax.bar(x - 0.2, by_hour_events, width=0.4, color="#bfbfbf", label="events")
    ax.bar(x + 0.2, by_hour_barks, width=0.4, color="#b11d1d", label="est. barks")
    ax.set_xlabel("hour of day (local)")
    ax.set_ylabel("count")
    ax.set_xticks(x[::3])
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_hour_hist.png", dpi=160)
    plt.close(fig)

    # ---- Figure 3: date x hour heatmap ----
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    vmax = max(int(heat.max()), 1)
    im = ax.imshow(heat, aspect="auto", cmap="Reds",
                   norm=LogNorm(vmin=1, vmax=vmax) if vmax > 1 else None)
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates)
    for tick, d in zip(ax.get_yticklabels(), dates):
        if d in flagged_dates:
            tick.set_color("#b11d1d")
            tick.set_fontweight("bold")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour of day (local)")
    ax.set_title("Estimated barks per (date, hour) — flagged cells outlined",
                 fontsize=11, pad=8)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("est. barks")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = int(heat[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=5.5,
                        color="white" if v > vmax / 2 else "black")
            if (i, j) in violation_cells:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="black",
                                       linewidth=1.1, zorder=5))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_date_hour.png", dpi=170)
    plt.close(fig)

    # ---- Figure 4: dow x hour heatmap ----
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    vmax4 = max(int(dow_hour.max()), 1)
    im4 = ax.imshow(dow_hour, aspect="auto", cmap="Reds",
                    norm=LogNorm(vmin=1, vmax=vmax4) if vmax4 > 1 else None)
    ax.set_yticks(range(7))
    ax.set_yticklabels(DOW_NAMES)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour of day (local)")
    cb4 = fig.colorbar(im4, ax=ax, pad=0.01)
    cb4.set_label("est. barks")
    for i in range(7):
        for j in range(24):
            v = int(dow_hour[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=6,
                        color="white" if v > vmax4 / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_dow_hour.png", dpi=170)
    plt.close(fig)

    print(f"[out] wrote 4 figures to {OUT_DIR}/")
    print(f"[summary] {total_events} events, {total_clips} bark clips, "
          f"{total_barks} est. barks across {len(dates)} days, "
          f"{len(flagged_dates)} days flagged")


if __name__ == "__main__":
    main()
