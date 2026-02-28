#!/usr/bin/env python3
"""
Reconstruct a waveform from its noisy version using a trained diffusion model.

Usage:
  python scripts/reconstruct_from_noisy.py --run_dir Catch-A-Waveform/outputs/we_are_the_champion_10 \
      --real_path Catch-A-Waveform/outputs/we_are_the_champion_10/real@4000Hz.wav \
      --t 0 --out recon_from_noisy.wav

If --real_path is omitted, the script picks the highest-resolution real@*Hz.wav
in the run directory. By default t=0 (no additional forward noise); increase t
to add more noise before denoising (0 <= t < diffusion_steps).
"""

import argparse
import glob
import os
import sys

# Ensure libsndfile is found on macOS/Homebrew setups BEFORE importing soundfile
os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

try:
    import numpy as np
    import soundfile as sf
    import torch
except Exception as exc:
    print(f"[setup] failed to import core deps: {exc}")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "Catch-A-Waveform")
sys.path.insert(0, PROJECT_ROOT)

from params import Params  # noqa: E402
from models import diffusion as diffusion_models  # noqa: E402
from utils.utils import build_diffusion_schedule, params_from_log  # noqa: E402


def _pick_real(run_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(run_dir, "real@*Hz.wav")))
    if not candidates:
        raise FileNotFoundError("No real@*Hz.wav found in run directory")
    # pick highest fs (last after numeric sort)
    def _key(p):
        name = os.path.basename(p)
        num = name.split("@")[1].replace("Hz.wav", "")
        try:
            return float(num)
        except ValueError:
            return 0

    return sorted(candidates, key=_key)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to run directory (contains log.txt and netDiffScale*.pth)")
    ap.add_argument("--real_path", default=None, help="Path to real waveform (defaults to highest real@*Hz.wav in run_dir)")
    ap.add_argument("--t", type=int, default=0, help="Diffusion timestep to use for forward noising (0 <= t < diffusion_steps)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for epsilon noise")
    ap.add_argument("--out", default="recon_from_noisy.wav", help="Output wav file (saved inside run_dir if relative)")
    ap.add_argument("--noise_scale", type=float, default=1.0, help="Multiplier on forward noise std (sqrt(1-alpha_bar))")
    args = ap.parse_args()

    run_dir = args.run_dir
    log_path = os.path.join(run_dir, "log.txt")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"log.txt not found in {run_dir}")

    params = params_from_log(log_path)
    params.device = torch.device("cpu")

    # choose real audio
    real_path = args.real_path or _pick_real(run_dir)
    real_np, sr = sf.read(real_path)
    params.Fs = sr
    if len(params.fs_list) == 0:
        params.fs_list = [sr]
    final_scale = min(len(params.fs_list) - 1, len(glob.glob(os.path.join(run_dir, "netDiffScale*.pth"))) - 1)

    # load model
    net_path = os.path.join(run_dir, f"netDiffScale{final_scale}.pth")
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"Missing model: {net_path}")
    params.current_fs = params.fs_list[final_scale]
    params.hidden_channels = (
        params.hidden_channels_init
        if final_scale == 0
        else int(params.hidden_channels_init * params.growing_hidden_channels_factor)
    )
    net = diffusion_models.DiffusionUNet1D(params)
    net.load_state_dict(torch.load(net_path, map_location="cpu"))
    net.eval()

    betas, alphas, acp, _ = build_diffusion_schedule(params, torch.device("cpu"))
    T = params.diffusion_steps
    t = max(0, min(args.t, T - 1))

    torch.manual_seed(args.seed)
    x0 = torch.tensor(real_np, dtype=torch.float32).view(1, 1, -1)
    eps = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(acp[t]).view(1, 1, 1)
    sqrt_1m = torch.sqrt(1 - acp[t]).view(1, 1, 1)
    x_t = sqrt_ab * x0 + args.noise_scale * sqrt_1m * eps

    eps_pred = net(x_t, torch.tensor([t]), torch.zeros_like(x_t))
    x0_pred = (x_t - sqrt_1m * eps_pred) / sqrt_ab
    x0_pred = torch.clamp(x0_pred, -1, 1)

    # Avoid torch->numpy bridge (can fail with NumPy 2.x + PyTorch)
    audio_np = np.array(x0_pred.squeeze().cpu().tolist(), dtype=np.float32)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(run_dir, out_path)
    sf.write(out_path, audio_np, sr, subtype="PCM_16")
    # diagnostics
    def snr(a, b):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        n = min(len(a), len(b))
        a = a[:n]; b = b[:n]
        err = a - b
        return 10 * np.log10(np.sum(a * a) / (np.sum(err * err) + 1e-12))

    x0_np = audio_np  # recon
    x_t_np = np.array(x_t.squeeze().cpu().tolist(), dtype=np.float32)
    x0_true = real_np[: len(x0_np)]
    print(f"Saved reconstruction to {out_path}")
    print(f"[diag] effective t={t}/{T-1}, noise_scale={args.noise_scale}")
    print(f"[diag] SNR(x0, x_t): {snr(x0_true, x_t_np):.2f} dB")
    print(f"[diag] SNR(x0, recon): {snr(x0_true, x0_np):.2f} dB")


if __name__ == "__main__":
    main()
