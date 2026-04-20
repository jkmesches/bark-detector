#!/usr/bin/env python3
"""
Score every recording with the trained CNN and produce a per-hour bark report.

Timestamps in filenames are UTC (bark_detector writes them with datetime.now(timezone.utc)).
This script converts them to America/Denver local time for the report.

Pipeline:
  CPU workers  → soundfile.read each WAV to a fixed-length float32 array (I/O-bound)
  GPU          → MelSpectrogram + power_to_db(ref=max) + normalize + CNN forward

Outputs:
  bark_report.csv    — per (date, hour) row: n_events, n_barks, mean_score  (local time)
  bark_report.png    — timeline, hour-of-day histogram, and date x hour heatmap  (local time)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from tqdm import tqdm

from label_barks import (
    THRESHOLD_PATH,
    EXCLUDE_DATES,
    PROJECT,
    REC_DIR,
    META_PATH,
    MODEL_PATH,
    bark_id_from_name,
)

CSV_OUT = PROJECT / "bark_report.csv"
PNG_OUT = PROJECT / "bark_report.png"
LOCAL_TZ = ZoneInfo("America/Denver")

# ---- worker state ----
_WORKER_SR: int | None = None
_WORKER_N: int | None = None


def _worker_init(sr: int, n_samples: int):
    global _WORKER_SR, _WORKER_N
    _WORKER_SR = sr
    _WORKER_N = n_samples


def _load_wav(wav_path_str: str) -> np.ndarray:
    """Load WAV as mono float32 at target SR, padded/truncated to fixed length."""
    import soundfile as sf
    y, sr = sf.read(wav_path_str, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != _WORKER_SR:
        # Rare fallback — recordings are 48 kHz natively.
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=_WORKER_SR).astype(np.float32)
    n = _WORKER_N
    if len(y) >= n:
        return y[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: len(y)] = y
    return out


def load_threshold(default: float) -> float:
    if THRESHOLD_PATH.exists():
        try:
            return float(json.loads(THRESHOLD_PATH.read_text()).get("threshold", default))
        except (json.JSONDecodeError, ValueError):
            pass
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="parallel CPU workers for WAV I/O")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override decision threshold (default: bark_cnn_threshold.json or 0.5)")
    ap.add_argument("--min-viz-score", type=float, default=0.1,
                    help="drop detector events below this score from all aggregates/plots (default 0.1)")
    ap.add_argument("--bark-min-gap-ms", type=float, default=150.0,
                    help="minimum spacing between distinct bark peaks within a clip")
    ap.add_argument("--bark-rel-db", type=float, default=12.0,
                    help="a peak counts if its RMS is within this many dB of the clip's loudest frame")
    ap.add_argument("--bark-floor-db", type=float, default=-45.0,
                    help="absolute dBFS floor for a peak to count")
    ap.add_argument("--bark-max-per-clip", type=int, default=8,
                    help="cap estimated barks per clip (sanity limit)")
    # Heuristics for Fort Collins § 4-94 (time of day, duration, noise level).
    # The ordinance sets no numeric thresholds — these are configurable defaults.
    ap.add_argument("--night-start", type=int, default=22, help="night hours start (local, inclusive)")
    ap.add_argument("--night-end", type=int, default=7, help="night hours end (local, exclusive)")
    ap.add_argument("--day-hour-barks", type=int, default=20,
                    help="barks/hour threshold during daytime to flag a violation")
    ap.add_argument("--night-hour-barks", type=int, default=5,
                    help="barks/hour threshold during nighttime to flag a violation")
    ap.add_argument("--sustained-hours", type=int, default=3,
                    help="count of consecutive hours meeting --sustained-min-barks to flag sustained-duration violation")
    ap.add_argument("--sustained-min-barks", type=int, default=5,
                    help="per-hour bark count that qualifies as part of a sustained run")
    ap.add_argument("--day-total-barks", type=int, default=100,
                    help="total barks in a single day that flags a daily violation")
    args = ap.parse_args()

    import torch
    import torchaudio
    from train_bark_cnn import BarkCNN

    if not META_PATH.exists() or not MODEL_PATH.exists():
        raise SystemExit("no model on disk — run train_bark_cnn.py")
    meta = json.loads(META_PATH.read_text())
    cfg = meta["audio"]
    norm = meta["normalize"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available — falling back to CPU (will be slow)")

    model = BarkCNN().to(device).eval()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    mel_xform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        win_length=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        f_min=cfg["f_min"],
        f_max=cfg["f_max"],
        power=2.0,
        center=True,
        pad_mode="constant",
        norm="slaney",
        mel_scale="slaney",
    ).to(device)

    norm_mean = torch.tensor(float(norm["mean"]), device=device)
    norm_std = torch.tensor(float(norm["std"]), device=device)

    threshold = args.threshold if args.threshold is not None else load_threshold(
        meta.get("default_threshold", 0.5)
    )
    print(f"[model] device={device}  threshold={threshold:.3f}  oof_acc={meta.get('oof_accuracy'):.3f}")

    all_wavs = sorted(REC_DIR.glob("bark_*.wav"))
    files: list[Path] = []
    for p in all_wavs:
        bid = bark_id_from_name(p.name)
        if bid.split("_")[0] in EXCLUDE_DATES:
            continue
        files.append(p)
    print(f"[data] scoring {len(files)} files (skipping {len(all_wavs) - len(files)} excluded)"
          f"  workers={args.workers}  batch={args.batch_size}")

    sr = cfg["sample_rate"]
    n_samples = int(sr * cfg["clip_seconds"])
    scores = np.zeros(len(files), dtype=np.float32)
    barks_per_clip = np.zeros(len(files), dtype=np.int16)
    B = args.batch_size

    # GPU bark-peak counter params
    env_win = max(int(0.020 * sr), 1)   # 20 ms RMS window
    env_hop = max(int(0.010 * sr), 1)   # 10 ms hop
    frames_per_sec = sr / env_hop
    min_gap_frames = max(int(args.bark_min_gap_ms * 1e-3 * frames_per_sec), 1)
    pool_k = min_gap_frames * 2 + 1

    import torch.nn.functional as F

    def count_barks_gpu(waves: "torch.Tensor") -> "torch.Tensor":
        # waves: (B, N). Returns int tensor (B,).
        sq = waves.pow(2).unsqueeze(1)
        env = F.avg_pool1d(sq, kernel_size=env_win, stride=env_hop).squeeze(1)
        env_db = 10.0 * torch.log10(env.clamp_min(1e-12))
        env_max = F.max_pool1d(env_db.unsqueeze(1), kernel_size=pool_k,
                               stride=1, padding=min_gap_frames).squeeze(1)
        is_local_max = env_db >= env_max - 1e-6
        clip_max = env_db.amax(dim=1, keepdim=True)
        above_rel = env_db > (clip_max - args.bark_rel_db)
        above_floor = env_db > args.bark_floor_db
        peak_mask = is_local_max & above_rel & above_floor
        counts = peak_mask.sum(dim=1).clamp(max=args.bark_max_per_clip)
        return counts.to(torch.int16)

    wave_buf: list[np.ndarray] = []
    idx_buf: list[int] = []
    amin = 1e-10

    def flush():
        if not wave_buf:
            return
        waves = torch.from_numpy(np.stack(wave_buf)).to(device, non_blocking=True)
        with torch.no_grad():
            S = mel_xform(waves)                                       # (B, mel, T)
            S_db = 10.0 * torch.log10(torch.clamp(S, min=amin))
            ref_db = 10.0 * torch.log10(torch.clamp(
                torch.amax(S, dim=(-2, -1), keepdim=True), min=amin))
            S_db = torch.clamp(S_db - ref_db, min=-80.0)
            S_norm = (S_db - norm_mean) / norm_std
            logits = model(S_norm.unsqueeze(1)).float()
            probs = torch.sigmoid(logits).cpu().numpy()
            counts = count_barks_gpu(waves).cpu().numpy()
        for j, idx in enumerate(idx_buf):
            scores[idx] = probs[j]
            barks_per_clip[idx] = counts[j]
        wave_buf.clear()
        idx_buf.clear()

    path_strs = [str(p) for p in files]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(sr, n_samples),
    ) as ex:
        with tqdm(total=len(files), desc="scoring", unit="clip", smoothing=0.05) as pbar:
            for i, wav in enumerate(ex.map(_load_wav, path_strs, chunksize=32)):
                wave_buf.append(wav)
                idx_buf.append(i)
                if len(wave_buf) >= B:
                    flush()
                pbar.update(1)
            flush()

    # Parse UTC timestamp from filename, convert to local time.
    timestamps: list[datetime | None] = []
    for p in files:
        try:
            ts_utc = datetime.strptime(bark_id_from_name(p.name), "%Y%m%d_%H%M%S")
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
            ts_local = ts_utc.astimezone(LOCAL_TZ)
        except ValueError:
            ts_local = None
        timestamps.append(ts_local)

    preds = (scores >= threshold).astype(np.int8)
    # Per-clip estimated barks: 0 for non-bark clips, else clamp to >= 1.
    barks_est = np.where(preds == 1, np.maximum(barks_per_clip, 1), 0).astype(np.int32)

    raw_total = int(scores.size)
    excluded_local = {f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in EXCLUDE_DATES}
    local_excluded_mask = np.array(
        [ts is not None and ts.date().isoformat() in excluded_local for ts in timestamps],
        dtype=bool,
    )
    keep_mask = (scores >= args.min_viz_score) & (~local_excluded_mask)
    dropped_low = int(((scores < args.min_viz_score) & (~local_excluded_mask)).sum())
    dropped_excluded = int(local_excluded_mask.sum())
    print(f"[filter] dropping {dropped_low} events with score < {args.min_viz_score:.2f} "
          f"from aggregates ({100*dropped_low/max(raw_total,1):.1f}%)")
    if dropped_excluded:
        print(f"[filter] dropping {dropped_excluded} events on excluded local dates "
              f"({sorted(excluded_local)})")

    bucket = defaultdict(lambda: {
        "n_events": 0, "n_bark_clips": 0, "n_barks_est": 0, "score_sum": 0.0})
    for ts, score, pred, est, keep in zip(
            timestamps, scores, preds, barks_est, keep_mask):
        if ts is None or not keep:
            continue
        key = (ts.date().isoformat(), ts.hour)
        b = bucket[key]
        b["n_events"] += 1
        b["n_bark_clips"] += int(pred)
        b["n_barks_est"] += int(est)
        b["score_sum"] += float(score)

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rows = []
    for (d, h), b in sorted(bucket.items()):
        dow = datetime.fromisoformat(d).weekday()
        rows.append({
            "date": d,
            "dow": dow_names[dow],
            "hour": h,
            "n_events": b["n_events"],
            "n_bark_clips": b["n_bark_clips"],
            "n_barks_est": b["n_barks_est"],
            "mean_score": round(b["score_sum"] / b["n_events"], 4),
        })
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "dow", "hour", "n_events",
            "n_bark_clips", "n_barks_est", "mean_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"[out] wrote {CSV_OUT}")

    total_events = int(keep_mask.sum())
    total_bark_clips = int(preds[keep_mask].sum())
    total_barks_est = int(barks_est[keep_mask].sum())
    mult = total_barks_est / max(total_bark_clips, 1)
    print(f"\n[summary] timezone: {LOCAL_TZ.key} (filenames are UTC, converted for report)")
    print(f"[summary] detector events (score >= {args.min_viz_score:.2f}): "
          f"{total_events} (of {raw_total} raw)")
    print(f"[summary] bark clips (score >= {threshold:.2f}): {total_bark_clips} "
          f"({100*total_bark_clips/max(total_events,1):.1f}% of shown events)")
    print(f"[summary] estimated total barks: {total_barks_est} "
          f"(avg {mult:.2f} barks per bark-clip)")
    print(f"[summary] likely false positives shown: {total_events - total_bark_clips}")

    by_hour = np.zeros(24, dtype=np.int64)            # estimated barks by hour
    by_hour_events = np.zeros(24, dtype=np.int64)
    dow_hour = np.zeros((7, 24), dtype=np.int64)      # estimated barks by (dow, hour)
    dow_hour_events = np.zeros((7, 24), dtype=np.int64)
    for ts, est, keep in zip(timestamps, barks_est, keep_mask):
        if ts is None or not keep:
            continue
        by_hour_events[ts.hour] += 1
        dow = ts.weekday()
        dow_hour_events[dow, ts.hour] += 1
        if est > 0:
            by_hour[ts.hour] += int(est)
            dow_hour[dow, ts.hour] += int(est)
    peak_hour = int(by_hour.argmax())
    print(f"[summary] peak bark hour of day (local): {peak_hour:02d}:00 "
          f"({by_hour[peak_hour]} est. barks)")

    dow_totals = dow_hour.sum(axis=1)
    peak_dow = int(dow_totals.argmax())
    flat_idx = int(dow_hour.argmax())
    peak_dow_h_dow, peak_dow_h_hour = divmod(flat_idx, 24)
    print(f"[summary] barks by day-of-week: "
          + ", ".join(f"{dow_names[i]}={int(dow_totals[i])}" for i in range(7)))
    print(f"[summary] peak day-of-week: {dow_names[peak_dow]} ({dow_totals[peak_dow]} barks)")
    print(f"[summary] peak (dow, hour): {dow_names[peak_dow_h_dow]} {peak_dow_h_hour:02d}:00 "
          f"({dow_hour[peak_dow_h_dow, peak_dow_h_hour]} barks)")

    # ------------------------------------------------------------------
    # Fort Collins § 4-94 heuristic flagging.
    # The ordinance judges "unreasonable noise" by (1) time of day,
    # (2) duration, (3) noise level, and (4) other disruptive factors,
    # with no numeric threshold. These heuristics approximate those
    # factors — they are NOT a legal finding.
    # ------------------------------------------------------------------
    def is_night(h: int) -> bool:
        return (h >= args.night_start) or (h < args.night_end)

    # violation_reasons[(date, hour)] -> set[str]
    violation_reasons: dict[tuple[str, int], set[str]] = defaultdict(set)
    day_reasons: dict[str, set[str]] = defaultdict(set)
    day_totals: dict[str, int] = defaultdict(int)

    # Build per-date hour arrays to detect sustained runs.
    by_date_hour_barks: dict[str, np.ndarray] = {}
    for r in rows:
        arr = by_date_hour_barks.setdefault(r["date"], np.zeros(24, dtype=np.int64))
        arr[r["hour"]] = r["n_barks_est"]
        day_totals[r["date"]] += r["n_barks_est"]

    for d, arr in by_date_hour_barks.items():
        for h in range(24):
            n = int(arr[h])
            if n == 0:
                continue
            if is_night(h) and n >= args.night_hour_barks:
                violation_reasons[(d, h)].add(f"night≥{args.night_hour_barks}")
                day_reasons[d].add("night-level")
            elif (not is_night(h)) and n >= args.day_hour_barks:
                violation_reasons[(d, h)].add(f"day≥{args.day_hour_barks}")
                day_reasons[d].add("day-level")

        # Sustained run: consecutive hours with >= sustained_min_barks.
        run_start = None
        for h in range(24):
            qualifies = arr[h] >= args.sustained_min_barks
            if qualifies and run_start is None:
                run_start = h
            if (not qualifies or h == 23) and run_start is not None:
                run_end = h if not qualifies else h + 1
                if run_end - run_start >= args.sustained_hours:
                    for hh in range(run_start, run_end):
                        violation_reasons[(d, hh)].add(
                            f"sustained≥{args.sustained_hours}h")
                    day_reasons[d].add("sustained")
                run_start = None

        if day_totals[d] >= args.day_total_barks:
            day_reasons[d].add(f"daily≥{args.day_total_barks}")

    # Attach violation reasons to CSV rows and rewrite.
    for r in rows:
        reasons = sorted(violation_reasons.get((r["date"], r["hour"]), set()))
        r["violation"] = ";".join(reasons)
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "dow", "hour", "n_events",
            "n_bark_clips", "n_barks_est", "mean_score", "violation"])
        w.writeheader()
        w.writerows(rows)
    print(f"[out] rewrote {CSV_OUT} with violation column")

    dates = sorted({r["date"] for r in rows})
    violating_days = sorted(d for d, reasons in day_reasons.items() if reasons)
    print(f"\n[ordinance] flagged days: {len(violating_days)} of {len(dates)}")
    for d in violating_days:
        dow = dow_names[datetime.fromisoformat(d).weekday()]
        reasons = sorted(day_reasons[d])
        print(f"  {d} ({dow})  total={day_totals[d]:>4}  reasons: {', '.join(reasons)}")

    date_to_row = {d: i for i, d in enumerate(dates)}
    heat = np.zeros((len(dates), 24), dtype=np.int64)
    for r in rows:
        heat[date_to_row[r["date"]], r["hour"]] = r["n_barks_est"]

    fig = plt.figure(figsize=(13, 15.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.4, 1.0], hspace=0.55,
                          top=0.90, bottom=0.04)

    criteria_lines = [
        "Fort Collins § 4-94 violation criteria (counts = estimated barks, not clips):",
        f"• night hours: {args.night_start:02d}:00–{args.night_end:02d}:00 local    "
        f"• sustained duration: ≥ {args.sustained_hours} consecutive hours with ≥ {args.sustained_min_barks} barks/hr",
        f"• hourly level: day ≥ {args.day_hour_barks} barks/hr    night ≥ {args.night_hour_barks} barks/hr    "
        f"• daily total ≥ {args.day_total_barks} barks",
        f"Flagged (date, hour) cells outlined in black; flagged dates in red on y-axis. "
        f"Events with score < {args.min_viz_score:.2f} excluded.",
    ]
    fig.text(0.5, 0.975, "\n".join(criteria_lines),
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff4c2",
                       edgecolor="#b8860b", linewidth=0.8))

    ax1 = fig.add_subplot(gs[0])
    hourly_dt, hourly_barks, hourly_events = [], [], []
    for r in rows:
        t = datetime.fromisoformat(r["date"]).replace(hour=r["hour"])
        hourly_dt.append(t)
        hourly_barks.append(r["n_barks_est"])
        hourly_events.append(r["n_events"])
    ax1.bar(hourly_dt, hourly_events, width=1 / 24, color="#cccccc", label="events", align="edge")
    ax1.bar(hourly_dt, hourly_barks, width=1 / 24, color="#b11d1d", label="est. barks", align="edge")
    ax1.set_title(f"Hourly bark activity — local time ({LOCAL_TZ.key})  "
                  f"(threshold={threshold:.2f}, {total_barks_est} est. barks over {len(dates)} days)")
    ax1.set_ylabel("count / hour")
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    x = np.arange(24)
    ax2.bar(x - 0.2, by_hour_events, width=0.4, color="#cccccc", label="events")
    ax2.bar(x + 0.2, by_hour, width=0.4, color="#b11d1d", label="est. barks")
    ax2.set_title(f"Totals by hour of day — local ({LOCAL_TZ.key})")
    ax2.set_xlabel("hour of day (local)")
    ax2.set_ylabel("count")
    ax2.set_xticks(x)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    vmax = max(int(heat.max()), 1)
    im = ax3.imshow(heat, aspect="auto", cmap="Reds",
                    norm=LogNorm(vmin=1, vmax=vmax) if vmax > 1 else None)
    ax3.set_yticks(range(len(dates)), dates)
    for tick, d in zip(ax3.get_yticklabels(), dates):
        if day_reasons.get(d):
            tick.set_color("#b11d1d")
            tick.set_fontweight("bold")
    ax3.set_xticks(range(24))
    ax3.set_xlabel("hour of day (local)")
    ax3.set_title(f"Barks per (date, hour) — local ({LOCAL_TZ.key})   "
                  f"[{len(violating_days)}/{len(dates)} days flagged under § 4-94 heuristics]")
    cb = fig.colorbar(im, ax=ax3)
    cb.set_label("est. barks")
    for i, d in enumerate(dates):
        for j in range(heat.shape[1]):
            if heat[i, j] > 0:
                ax3.text(j, i, str(heat[i, j]), ha="center", va="center",
                         fontsize=7, color="white" if heat[i, j] > vmax / 2 else "black")
            if violation_reasons.get((d, j)):
                ax3.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="black",
                                        linewidth=1.6, zorder=5))

    ax4 = fig.add_subplot(gs[3])
    vmax4 = max(int(dow_hour.max()), 1)
    im4 = ax4.imshow(dow_hour, aspect="auto", cmap="Reds",
                     norm=LogNorm(vmin=1, vmax=vmax4) if vmax4 > 1 else None)
    ax4.set_yticks(range(7), dow_names)
    ax4.set_xticks(range(24))
    ax4.set_xlabel("hour of day (local)")
    ax4.set_title(f"Barks per (day-of-week, hour) — local ({LOCAL_TZ.key})")
    cb4 = fig.colorbar(im4, ax=ax4)
    cb4.set_label("est. barks")
    for i in range(7):
        for j in range(24):
            if dow_hour[i, j] > 0:
                ax4.text(j, i, str(dow_hour[i, j]), ha="center", va="center",
                         fontsize=7, color="white" if dow_hour[i, j] > vmax4 / 2 else "black")

    fig.savefig(PNG_OUT, dpi=120)
    plt.close(fig)
    print(f"[out] wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
