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

    keys = sorted(set(real_map) & set(fake_map), key=_scale_key)
    pairs = [(k, real_map[k], fake_map[k]) for k in keys]
    return pairs


def _scale_key(key):
    if key.endswith('Hz'):
        num = key[:-2]
        try:
            return float(num)
        except ValueError:
            return key
    try:
        return float(key)
    except ValueError:
        return key


def _align(ref, est):
    min_len = min(len(ref), len(est))
    return ref[:min_len], est[:min_len]


def _normalize(ref, est):
    max_amp = max(np.max(np.abs(ref)), 1e-12)
    return ref / max_amp, est / max_amp


def _normalize_pair(a, b):
    max_amp = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-12)
    return a / max_amp, b / max_amp


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

    coh_plot = np.maximum(coh, 1e-6)
    axes[2].plot(coh_freqs, coh_plot)
    axes[2].set_title('Spectral Coherence')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Coherence')
    axes[2].set_yscale('log')
    axes[2].set_ylim(1e-6, 1.05)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_lsd_curve(out_path, freqs, lsd_curve):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freqs, lsd_curve)
    ax.set_title('LSD vs Frequency')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('LSD')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_psd_curve(out_path, psd_freqs, psd_ref, psd_est):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    psd_ref_db = 10 * np.log10(psd_ref + 1e-12)
    psd_est_db = 10 * np.log10(psd_est + 1e-12)
    ax.plot(psd_freqs, psd_ref_db, label='Real')
    ax.plot(psd_freqs, psd_est_db, label='Fake')
    ax.set_title('PSD (Welch)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_coherence_curve(out_path, coh_freqs, coh):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    coh_plot = np.maximum(coh, 1e-6)
    ax.plot(coh_freqs, coh_plot)
    ax.set_title('Spectral Coherence')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Coherence')
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_psd_compare(out_path, psd_freqs, psd_a, psd_b, label_a, label_b):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    psd_a_db = 10 * np.log10(psd_a + 1e-12)
    psd_b_db = 10 * np.log10(psd_b + 1e-12)
    ax.plot(psd_freqs, psd_a_db, label=label_a)
    ax.plot(psd_freqs, psd_b_db, label=label_b)
    ax.set_title('PSD (Welch)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_snr_by_scale(out_path, keys, snr_values, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(keys, snr_values, marker='o', color='red')
    ax.set_title(title)
    ax.set_xlabel('Downsampling scale (Hz)')
    ax.set_ylabel('SNR (dB)')
    ax.set_xticks(keys)
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
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


def _plot_spectrogram(out_path, signal_data, sr, n_fft, hop_length, title):
    import matplotlib.pyplot as plt

    stft = librosa.stft(signal_data, n_fft=n_fft, hop_length=hop_length)
    mag_db = librosa.amplitude_to_db(np.abs(stft) + 1e-12, ref=np.max)
    time_axis = librosa.frames_to_time(np.arange(mag_db.shape[1]), sr=sr, hop_length=hop_length)
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        mag_db,
        origin='lower',
        aspect='auto',
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    )
    ax.set_title(title)
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
    parser.add_argument('--reconstructed_glob', help='Glob for reconstructed signals (optional)')
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
    psd_reconstructed_curves = []
    psd_fake_for_reconstructed_curves = []
    summary_psd_rec_freqs = None
    snr_reconstructed_by_scale = []

    reconstructed_map = None
    reconstructed_key_max = None
    if args.reconstructed_glob:
        reconstructed_files = glob.glob(args.reconstructed_glob)
        reconstructed_map = {
            os.path.splitext(os.path.basename(p))[0].split('@', 1)[1]: p
            for p in reconstructed_files
            if '@' in os.path.basename(p)
        }

    max_scale_key = None
    if pairs:
        max_scale_key = max((k for k, _, _ in pairs), key=_scale_key)

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

            if reconstructed_map is not None and key in reconstructed_map:
                if reconstructed_key_max is None:
                    common_keys = set(reconstructed_map) & {k for k, _, _ in pairs}
                    if common_keys:
                        reconstructed_key_max = max(common_keys, key=_scale_key)
                if key == reconstructed_key_max:
                    rec, rec_sr = _load_audio(reconstructed_map[key])
                    if rec_sr != ref_sr:
                        rec = librosa.resample(rec, orig_sr=rec_sr, target_sr=ref_sr)
                    if args.normalize:
                        est, rec = _normalize_pair(est, rec)
                    est_rec, rec = _align(est, rec)
                    psd_rec_freqs, psd_fake_for_rec = signal.welch(est_rec, fs=ref_sr, nperseg=args.nperseg)
                    _, psd_rec = signal.welch(rec, fs=ref_sr, nperseg=args.nperseg)
                    _write_curve(base + '_psd_reconstructed.csv', psd_rec_freqs, psd_rec, 'psd_reconstructed')

                    if summary_psd_rec_freqs is None:
                        summary_psd_rec_freqs = psd_rec_freqs
                    psd_fake_for_reconstructed_curves.append(
                        _interp_to_grid(summary_psd_rec_freqs, psd_rec_freqs, psd_fake_for_rec)
                    )
                    psd_reconstructed_curves.append(
                        _interp_to_grid(summary_psd_rec_freqs, psd_rec_freqs, psd_rec)
                    )

                    if not args.no_plots:
                        _plot_psd_compare(
                            base + '_psd_fake_vs_reconstructed.png',
                            psd_rec_freqs,
                            psd_fake_for_rec,
                            psd_rec,
                            'Fake',
                            'Reconstructed',
                        )

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
                _plot_lsd_curve(base + '_lsd.png', freqs, lsd_curve)
                _plot_psd_curve(base + '_psd.png', psd_freqs, psd_ref, psd_est)
                _plot_coherence_curve(base + '_coherence.png', coh_freqs, coh)
                diff_db = np.abs(log_ref - log_est)
                _plot_stft_diff(base + '_stft_diff.png', diff_db, ref_sr, args.hop_length)

            if not args.no_plots and key == max_scale_key:
                _plot_spectrogram(
                    base + '_real_spectrogram.png',
                    ref,
                    ref_sr,
                    args.n_fft,
                    args.hop_length,
                    'Real Spectrogram (max scale)',
                )
                _plot_spectrogram(
                    base + '_fake_spectrogram.png',
                    est,
                    ref_sr,
                    args.n_fft,
                    args.hop_length,
                    'Fake Spectrogram (max scale)',
                )

            if reconstructed_map is not None and key in reconstructed_map:
                rec, rec_sr = _load_audio(reconstructed_map[key])
                if rec_sr != ref_sr:
                    rec = librosa.resample(rec, orig_sr=rec_sr, target_sr=ref_sr)
                if args.normalize:
                    ref, rec = _normalize_pair(ref, rec)
                ref_rec, rec = _align(ref, rec)
                snr_rec = calc_snr(rec, ref_rec)
                snr_reconstructed_by_scale.append((key, snr_rec))

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

    if psd_reconstructed_curves:
        mean_psd_fake_for_rec = np.mean(np.vstack(psd_fake_for_reconstructed_curves), axis=0)
        mean_psd_rec = np.mean(np.vstack(psd_reconstructed_curves), axis=0)
        _write_curve(
            os.path.join(args.out_dir, 'summary_psd_reconstructed.csv'),
            summary_psd_rec_freqs,
            mean_psd_rec,
            'psd_reconstructed',
        )
        _write_curve(
            os.path.join(args.out_dir, 'summary_psd_fake_for_reconstructed.csv'),
            summary_psd_rec_freqs,
            mean_psd_fake_for_rec,
            'psd_fake',
        )
        if not args.no_plots:
            _plot_psd_compare(
                os.path.join(args.out_dir, 'summary_psd_fake_vs_reconstructed.png'),
                summary_psd_rec_freqs,
                mean_psd_fake_for_rec,
                mean_psd_rec,
                'Fake',
                'Reconstructed',
            )

    if snr_reconstructed_by_scale:
        snr_reconstructed_by_scale = sorted(snr_reconstructed_by_scale, key=lambda x: _scale_key(x[0]))
        snr_curve_path = os.path.join(args.out_dir, 'snr_reconstructed_by_scale.csv')
        with open(snr_curve_path, 'w', encoding='utf-8') as f:
            f.write('scale_key,snr_db\n')
            for key, snr_val in snr_reconstructed_by_scale:
                f.write('%s,%.6f\n' % (key, snr_val))
        if not args.no_plots:
            keys = [k for k, _ in snr_reconstructed_by_scale]
            snr_vals = [v for _, v in snr_reconstructed_by_scale]
            _plot_snr_by_scale(
                os.path.join(args.out_dir, 'snr_reconstructed_by_scale.png'),
                keys,
                snr_vals,
                'Reconstructed vs Real SNR by Scale',
            )


if __name__ == '__main__':
    main()


# python scripts/eval_freq_curves.py --real_glob "Catch-A-Waveform/outputs/_exp_crops/we_are_the_champion_5/real@*Hz.wav" --fake_glob "Catch-A-Waveform/outputs/_exp_crops/we_are_the_champion_5/fake@*Hz.wav" --reconstructed_glob "Catch-A-Waveform/outputs/_exp_crops/we_are_the_champion_5/reconstructed@*Hz.wav" --out_dir "Catch-A-Waveform/outputs/metrics/we_are_the_champion_5"
