import argparse
import os
import sys

import librosa
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', 'Catch-A-Waveform')

sys.path.insert(0, PROJECT_ROOT)

from utils.utils import calc_snr, calc_lsd  # noqa: E402


def _safe_import_pesq():
    try:
        from pesq import pesq  # type: ignore
    except Exception:
        return None
    return pesq


def _safe_import_stoi():
    try:
        from pystoi.stoi import stoi  # type: ignore
    except Exception:
        return None
    return stoi


def _load_audio(path):
    signal, sr = librosa.load(path, sr=None, mono=True)
    return signal, sr


def _align_signals(ref, est):
    min_len = min(len(ref), len(est))
    return ref[:min_len], est[:min_len]


def _maybe_normalize(ref, est):
    max_amp = max(np.max(np.abs(ref)), 1e-12)
    return ref / max_amp, est / max_amp


def main():
    parser = argparse.ArgumentParser(description='Compute objective audio metrics.')
    parser.add_argument('--ref', required=True, help='Path to reference audio (wav)')
    parser.add_argument('--est', required=True, help='Path to estimated audio (wav)')
    parser.add_argument('--normalize', action='store_true', help='Normalize by ref max abs before metrics')
    parser.add_argument('--pesq', action='store_true', help='Compute PESQ if pesq is installed')
    parser.add_argument('--stoi', action='store_true', help='Compute STOI if pystoi is installed')

    args = parser.parse_args()

    ref, ref_sr = _load_audio(args.ref)
    est, est_sr = _load_audio(args.est)

    if ref_sr != est_sr:
        est = librosa.resample(est, orig_sr=est_sr, target_sr=ref_sr)
        est_sr = ref_sr

    if args.normalize:
        ref, est = _maybe_normalize(ref, est)

    ref, est = _align_signals(ref, est)

    snr = calc_snr(est, ref)
    lsd = calc_lsd(est, ref)
    print('SNR: %.2f dB' % snr)
    print('LSD: %.2f' % lsd)

    if args.pesq:
        pesq_fn = _safe_import_pesq()
        if pesq_fn is None:
            print('PESQ: skipped (install "pesq")')
        else:
            if ref_sr not in (8000, 16000):
                ref_16k = librosa.resample(ref, orig_sr=ref_sr, target_sr=16000)
                est_16k = librosa.resample(est, orig_sr=ref_sr, target_sr=16000)
                pesq_sr = 16000
            else:
                ref_16k = ref
                est_16k = est
                pesq_sr = ref_sr
            mode = 'wb' if pesq_sr == 16000 else 'nb'
            pesq_val = pesq_fn(pesq_sr, ref_16k, est_16k, mode)
            print('PESQ (%s): %.3f' % (mode, pesq_val))

    if args.stoi:
        stoi_fn = _safe_import_stoi()
        if stoi_fn is None:
            print('STOI: skipped (install "pystoi")')
        else:
            stoi_val = stoi_fn(ref, est, ref_sr, extended=False)
            print('STOI: %.3f' % stoi_val)


if __name__ == '__main__':
    main()

# Commands : 
# python3 scripts/eval_metrics.py --ref Catch-A-Waveform/outputs/we_are_the_champion_5/real@2000Hz.wav --est Catch-A-Waveform/outputs/we_are_the_champion_5/fake@2000Hz.wav --pesq --stoi
