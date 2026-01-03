#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import soundfile as sf

import numpy as np


@dataclass
class CropSpec:
    length_sec: float
    crop_idx: int
    start_sample: int
    start_sec: float
    rms: float
    out_wav: Path

def compute_rms(x: np.ndarray) -> float:
    # x: shape (n,) or (n, c)
    x = np.asarray(x)
    if x.ndim == 2:
        x_mono = np.mean(x, axis=1)
    else:
        x_mono = x
    return float(np.sqrt(np.mean(np.square(x_mono), dtype=np.float64) + 1e-12))


def pick_starts(
    total_samples: int,
    crop_samples: int,
    n_crops: int,
    mode: str,
    seed: int,
) -> List[int]:
    """Return list of start indices (in samples)."""
    rng = random.Random(seed)

    max_start = total_samples - crop_samples
    if max_start < 0:
        return []

    if mode == "random":
        # random distinct-ish starts
        starts = [rng.randint(0, max_start) for _ in range(n_crops)]
        return starts

    if mode == "stratified":
        # centers at 10%,30%,50%,70%,90% then jitter slightly
        # if n_crops != 5, we generalize evenly spaced percentiles
        perc = np.linspace(0.1, 0.9, n_crops)
        starts = []
        jitter = int(0.02 * total_samples)  # 2% jitter
        for p in perc:
            center = int(p * total_samples)
            s = center - crop_samples // 2
            s += rng.randint(-jitter, jitter)
            s = max(0, min(max_start, s))
            starts.append(s)
        return starts

    raise ValueError(f"Unknown mode: {mode}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_train(
    caw_dir: Path,
    input_file_name: str,
    extra_args: Optional[str],
    env: Optional[dict] = None,
) -> Tuple[int, float]:
    """
    Launch: python train_main.py --input_file <name> [extra_args]
    Returns (returncode, elapsed_sec)
    """
    cmd = [sys.executable, "train_main.py", "--input_file", input_file_name]
    if extra_args:
        # split like a shell would, but without shell=True
        import shlex
        cmd.extend(shlex.split(extra_args))

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(caw_dir), env=env)
    elapsed = time.time() - t0
    return proc.returncode, float(elapsed)


def main():
    ap = argparse.ArgumentParser(
        description="CAW experiment: vary training sample length by cropping input audio and training on each crop."
    )
    ap.add_argument("--caw_dir", default="Catch-A-Waveform", help="Path to Catch-A-Waveform repo")
    ap.add_argument("--file_name", default="oiseaux.wav", help="Original wav in <caw_dir>/inputs/")
    ap.add_argument(
        "--lengths_sec",
        default="1,2,5,10,20,40",
        help="Comma-separated crop lengths in seconds (e.g. 1,2,5,10)",
    )
    ap.add_argument("--n_crops", type=int, default=5, help="Number of crops per length")
    ap.add_argument("--mode", choices=["random", "stratified"], default="stratified", help="How to pick crop positions")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (affects crop positions)")
    ap.add_argument(
        "--min_rms",
        type=float,
        default=0.01,
        help="Reject crops with RMS below this threshold (helps avoid silence)",
    )
    ap.add_argument(
        "--max_tries_per_crop",
        type=int,
        default=20,
        help="How many times to retry a crop if it's below min_rms",
    )
    ap.add_argument(
        "--tmp_dir_name",
        default="_exp_crops",
        help="Subfolder inside inputs/ to store temporary cropped wavs",
    )
    ap.add_argument(
        "--log_csv",
        default="caw_len_experiment_log.csv",
        help="CSV log path (written next to this script by default)",
    )
    ap.add_argument(
        "--keep_wavs",
        action="store_true",
        help="If set, do not delete temporary crop wavs at the end.",
    )
    ap.add_argument(
        "--extra_args",
        default="",
        help='Extra args appended to train_main.py, e.g. --extra_args "--epochs 200 --lr 1e-4"',
    )

    args = ap.parse_args()

    caw_dir = Path(args.caw_dir).resolve()
    inputs_dir = caw_dir / "inputs"
    src_wav = inputs_dir / args.file_name
    tmp_dir = inputs_dir / args.tmp_dir_name
    log_csv = Path(args.log_csv).resolve()

    if not caw_dir.is_dir():
        raise SystemExit(f"❌ CAW directory not found: {caw_dir}")
    if not (caw_dir / "train_main.py").is_file():
        raise SystemExit(f"❌ train_main.py not found in: {caw_dir}")
    if not src_wav.is_file():
        raise SystemExit(f"❌ Audio file not found: {src_wav}")

    ensure_dir(tmp_dir)

    # Load audio
    x, sr = sf.read(str(src_wav), always_2d=False)
    total_samples = x.shape[0] if x.ndim == 1 else x.shape[0]
    total_sec = total_samples / sr
    print(f"🎵 Loaded {src_wav.name}: sr={sr}, duration={total_sec:.2f}s, samples={total_samples}")

    lengths = []
    for tok in args.lengths_sec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        lengths.append(float(tok))

    # Prepare CSV log
    is_new = not log_csv.exists()
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "timestamp",
                "src_file",
                "sr",
                "length_sec",
                "crop_idx",
                "start_sec",
                "start_sample",
                "rms",
                "crop_file",
                "train_returncode",
                "train_elapsed_sec",
            ])

        # Main loop
        for L_sec in lengths:
            crop_samples = int(round(L_sec * sr))
            if crop_samples <= 0:
                print(f"⚠️ Skipping invalid length: {L_sec}")
                continue
            if crop_samples > total_samples:
                print(f"⚠️ Skipping length {L_sec}s > audio duration {total_sec:.2f}s")
                continue

            # Candidate starts (initial plan)
            starts = pick_starts(
                total_samples=total_samples,
                crop_samples=crop_samples,
                n_crops=args.n_crops,
                mode=args.mode,
                seed=args.seed + int(L_sec * 1000),
            )

            print("==============================")
            print(f"🧪 Length = {L_sec:.2f}s ({crop_samples} samples) | crops={len(starts)} | mode={args.mode}")

            for i, start in enumerate(starts):
                # Retry logic to avoid silent crops
                chosen_start = None
                chosen_rms = None

                rng = random.Random(args.seed + 10_000 + i + int(L_sec * 1000))
                max_start = total_samples - crop_samples

                for t in range(args.max_tries_per_crop):
                    s = start if t == 0 else rng.randint(0, max_start)
                    crop = x[s : s + crop_samples]
                    r = compute_rms(crop)
                    if r >= args.min_rms:
                        chosen_start, chosen_rms = s, r
                        break

                if chosen_start is None:
                    # log as skipped
                    print(f"⚠️  Crop {i}: could not find non-silent segment after {args.max_tries_per_crop} tries.")
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        src_wav.name,
                        sr,
                        L_sec,
                        i,
                        "",
                        "",
                        "",
                        "",
                        "SKIPPED",
                        "",
                    ])
                    f.flush()
                    continue

                start_sec = chosen_start / sr
                out_name = f"{src_wav.stem}__L{L_sec:.2f}s__i{i:02d}__s{start_sec:.2f}.wav"
                out_wav = tmp_dir / out_name

                # Write crop wav
                sf.write(str(out_wav), x[chosen_start : chosen_start + crop_samples], sr)
                print(f"🧩 Crop {i}: start={start_sec:.2f}s rms={chosen_rms:.4f} -> {out_wav.relative_to(inputs_dir)}")

                # Train on this crop
                rc, elapsed = run_train(
                    caw_dir=caw_dir,
                    input_file_name=str(out_wav.relative_to(inputs_dir)),  # path relative to inputs/
                    extra_args=args.extra_args.strip() or None,
                )

                print(f"train_main.py returncode={rc} elapsed={elapsed:.1f}s")

                # Log
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    src_wav.name,
                    sr,
                    L_sec,
                    i,
                    f"{start_sec:.6f}",
                    chosen_start,
                    f"{chosen_rms:.8f}",
                    str(out_wav),
                    rc,
                    f"{elapsed:.6f}",
                ])
                f.flush()

    if not args.keep_wavs:
        print(f"\n🧹 Cleaning temporary crops: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n✅ Done. Log written to: {log_csv}")


if __name__ == "__main__":
    main()

# python scripts/variation_len_train.py --file_name oiseaux.wav --lengths_sec 1,2,5,10,20 --n_crops 5 --mode stratified --min_rms 0.01 --extra_args "" 