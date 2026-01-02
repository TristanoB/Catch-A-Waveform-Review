import argparse
import glob
import os
import sys

import librosa
import numpy as np
from scipy import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', 'Catch-A-Waveform')
sys.path.insert(0, PROJECT_ROOT)

from utils.utils import calc_snr, calc_lsd  # noqa: E402


def _load_audio(path):
    signal_data, sr = librosa.load(path, sr=None, mono=True)
    return signal_data, sr


def _pair_files(real_glob, fake_glob):
    real_files = glob.glob(real_glob)
    fake_files = glob.glob(fake_glob)
    if not real_files:
        raise ValueError('No real files matched: %s' % real_glob)
    if not fake_files:
        raise ValueError('No fake files matched: %s' % fake_glob)

    def key_for(path):
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        if '@' in name:
            return name.split('@', 1)[1]
        return name

    real_map = {key_for(p): p for p in real_files}
    fake_map = {key_for(p): p for p in fake_files}

    keys = sorted(set(real_map) & set(fake_map), key=lambda k: float(k) if k.isdigit() else k)
    pairs = [(k, real_map[k], fake_map[k]) for k in keys]
    return pairs


def _align(ref, est):
    min_len = min(len(ref), len(est))
    return ref[:min_len], est[:min_len]


def _normalize(ref, est):
    max_amp = max(np.max(np.abs(ref)), 1e-12)
    return ref / max_amp, est / max_amp


def _plot_curves(out_path, freqs, lsd_curve, psd_freqs, psd_ref, psd_est, coh_freqs, coh):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 12))

    axes[0].plot(freqs, lsd_curve)
    axes[0].set_title('LSD vs Frequency')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('LSD')
    axes[0].grid(True, alpha=0.3)

    psd_ref_db = 10 * np.log10(psd_ref + 1e-12)
    psd_est_db = 10 * np.log10(psd_est + 1e-12)
    axes[1].plot(psd_freqs, psd_ref_db, label='Real')
    axes[1].plot(psd_freqs, psd_est_db, label='Fake')
    axes[1].set_title('PSD (Welch)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power (dB)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(coh_freqs, coh)
    axes[2].set_title('Spectral Coherence')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Coherence')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_stft_diff(out_path, diff_db, sr, hop_length):
    import matplotlib.pyplot as plt

    time_axis = librosa.frames_to_time(np.arange(diff_db.shape[1]), sr=sr, hop_length=hop_length)
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=(diff_db.shape[0] - 1) * 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        diff_db,
        origin='lower',
        aspect='auto',
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    )
    ax.set_title('STFT |log-mag diff|')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_curve(path, freqs, values, header_name):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('freq_hz,%s\n' % header_name)
        for freq, val in zip(freqs, values):
            f.write('%.6f,%.6f\n' % (freq, val))


def _interp_to_grid(target_freqs, source_freqs, values):
    if np.array_equal(target_freqs, source_freqs):
        return values
    return np.interp(target_freqs, source_freqs, values)


def main():
    parser = argparse.ArgumentParser(description='Evaluate frequency-aware metrics for CAW outputs.')
    parser.add_argument('--real_glob', required=True, help='Glob for real signals')
    parser.add_argument('--fake_glob', required=True, help='Glob for fake signals')
    parser.add_argument('--out_dir', default='outputs/metrics', help='Output directory')
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--nperseg', type=int, default=2048)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--no_plots', action='store_true')

    args = parser.parse_args()

    pairs = _pair_files(args.real_glob, args.fake_glob)
    os.makedirs(args.out_dir, exist_ok=True)

    summary_path = os.path.join(args.out_dir, 'summary.csv')
    summary_lsd_freqs = None
    summary_psd_freqs = None
    summary_coh_freqs = None
    lsd_curves = []
    psd_real_curves = []
    psd_fake_curves = []
    coh_curves = []

    with open(summary_path, 'w', encoding='utf-8') as summary:
        summary.write('key,real_path,fake_path,sr,snr_db,lsd\n')

        for key, real_path, fake_path in pairs:
            ref, ref_sr = _load_audio(real_path)
            est, est_sr = _load_audio(fake_path)

            if ref_sr != est_sr:
                est = librosa.resample(est, orig_sr=est_sr, target_sr=ref_sr)
                est_sr = ref_sr

            if args.normalize:
                ref, est = _normalize(ref, est)

            ref, est = _align(ref, est)

            snr = calc_snr(est, ref)
            lsd = calc_lsd(est, ref)

            summary.write('%s,%s,%s,%d,%.6f,%.6f\n' % (key, real_path, fake_path, ref_sr, snr, lsd))

            stft_ref = librosa.stft(ref, n_fft=args.n_fft, hop_length=args.hop_length)
            stft_est = librosa.stft(est, n_fft=args.n_fft, hop_length=args.hop_length)
            mag_ref = np.abs(stft_ref) + 1e-12
            mag_est = np.abs(stft_est) + 1e-12
            log_ref = np.log(mag_ref)
            log_est = np.log(mag_est)

            lsd_curve = np.sqrt(np.mean((log_ref - log_est) ** 2, axis=1))
            freqs = librosa.fft_frequencies(sr=ref_sr, n_fft=args.n_fft)

            psd_freqs, psd_ref = signal.welch(ref, fs=ref_sr, nperseg=args.nperseg)
            _, psd_est = signal.welch(est, fs=ref_sr, nperseg=args.nperseg)

            coh_freqs, coh = signal.coherence(ref, est, fs=ref_sr, nperseg=args.nperseg)

            base = os.path.join(args.out_dir, str(key))
            _write_curve(base + '_lsd.csv', freqs, lsd_curve, 'lsd')
            _write_curve(base + '_psd_real.csv', psd_freqs, psd_ref, 'psd_real')
            _write_curve(base + '_psd_fake.csv', psd_freqs, psd_est, 'psd_fake')
            _write_curve(base + '_coherence.csv', coh_freqs, coh, 'coherence')

            if summary_lsd_freqs is None:
                summary_lsd_freqs = freqs
                summary_psd_freqs = psd_freqs
                summary_coh_freqs = coh_freqs
            lsd_curves.append(_interp_to_grid(summary_lsd_freqs, freqs, lsd_curve))
            psd_real_curves.append(_interp_to_grid(summary_psd_freqs, psd_freqs, psd_ref))
            psd_fake_curves.append(_interp_to_grid(summary_psd_freqs, psd_freqs, psd_est))
            coh_curves.append(_interp_to_grid(summary_coh_freqs, coh_freqs, coh))

            if not args.no_plots:
                _plot_curves(base + '_curves.png', freqs, lsd_curve, psd_freqs, psd_ref, psd_est, coh_freqs, coh)
                diff_db = np.abs(log_ref - log_est)
                _plot_stft_diff(base + '_stft_diff.png', diff_db, ref_sr, args.hop_length)

    if lsd_curves:
        mean_lsd = np.mean(np.vstack(lsd_curves), axis=0)
        mean_psd_real = np.mean(np.vstack(psd_real_curves), axis=0)
        mean_psd_fake = np.mean(np.vstack(psd_fake_curves), axis=0)
        mean_coh = np.mean(np.vstack(coh_curves), axis=0)

        _write_curve(os.path.join(args.out_dir, 'summary_lsd.csv'), summary_lsd_freqs, mean_lsd, 'lsd')
        _write_curve(os.path.join(args.out_dir, 'summary_psd_real.csv'), summary_psd_freqs, mean_psd_real, 'psd_real')
        _write_curve(os.path.join(args.out_dir, 'summary_psd_fake.csv'), summary_psd_freqs, mean_psd_fake, 'psd_fake')
        _write_curve(os.path.join(args.out_dir, 'summary_coherence.csv'), summary_coh_freqs, mean_coh, 'coherence')

        if not args.no_plots:
            _plot_curves(
                os.path.join(args.out_dir, 'summary_curves.png'),
                summary_lsd_freqs,
                mean_lsd,
                summary_psd_freqs,
                mean_psd_real,
                mean_psd_fake,
                summary_coh_freqs,
                mean_coh,
            )


if __name__ == '__main__':
    main()
