import argparse
import glob
import os
import re
import sys

import librosa
import numpy as np
from scipy import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', 'Catch-A-Waveform')
sys.path.insert(0, PROJECT_ROOT)

from utils.utils import calc_snr, calc_lsd  # noqa: E402


LENGTH_RE = re.compile(r'__L(?P<length>[0-9]+(?:\.[0-9]+)?)')


def _load_audio(path):
    data, sr = librosa.load(path, sr=None, mono=True)
    return data, sr


def _align(ref, est):
    min_len = min(len(ref), len(est))
    return ref[:min_len], est[:min_len]


def _normalize(ref, est):
    max_amp = max(np.max(np.abs(ref)), 1e-12)
    return ref / max_amp, est / max_amp


def _key_for(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    if '@' in name:
        return name.split('@', 1)[1]
    return name


def _pair_files(real_glob, fake_glob):
    real_files = glob.glob(real_glob)
    fake_files = glob.glob(fake_glob)
    if not real_files or not fake_files:
        return []
    real_map = { _key_for(p): p for p in real_files }
    fake_map = { _key_for(p): p for p in fake_files }
    keys = sorted(set(real_map) & set(fake_map), key=_scale_key)
    return [(k, real_map[k], fake_map[k]) for k in keys]


def _scale_key(key):
    # "16000Hz" -> 16000.0 for sorting, fall back to string
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


def _parse_length(folder_name):
    match = LENGTH_RE.search(folder_name)
    if not match:
        return None
    return float(match.group('length'))


def _interp_to_grid(target_freqs, source_freqs, values):
    if np.array_equal(target_freqs, source_freqs):
        return values
    return np.interp(target_freqs, source_freqs, values)


def _compute_curves(ref, est, sr, n_fft, hop_length, nperseg):
    stft_ref = librosa.stft(ref, n_fft=n_fft, hop_length=hop_length)
    stft_est = librosa.stft(est, n_fft=n_fft, hop_length=hop_length)
    mag_ref = np.abs(stft_ref) + 1e-12
    mag_est = np.abs(stft_est) + 1e-12
    log_ref = np.log(mag_ref)
    log_est = np.log(mag_est)

    lsd_curve = np.sqrt(np.mean((log_ref - log_est) ** 2, axis=1))
    lsd_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    psd_freqs, psd_ref = signal.welch(ref, fs=sr, nperseg=nperseg)
    _, psd_est = signal.welch(est, fs=sr, nperseg=nperseg)

    coh_freqs, coh = signal.coherence(ref, est, fs=sr, nperseg=nperseg)

    return lsd_freqs, lsd_curve, psd_freqs, psd_ref, psd_est, coh_freqs, coh


def _load_and_align(real_path, est_path, normalize):
    ref, ref_sr = _load_audio(real_path)
    est, est_sr = _load_audio(est_path)

    if ref_sr != est_sr:
        est = librosa.resample(est, orig_sr=est_sr, target_sr=ref_sr)
        est_sr = ref_sr

    if normalize:
        ref, est = _normalize(ref, est)

    ref, est = _align(ref, est)
    return ref, est, ref_sr


def _plot_curves(out_path, freqs, lsd_curve, psd_freqs, psd_ref, psd_est, coh_freqs, coh, est_label):
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
    axes[1].plot(psd_freqs, psd_est_db, label=est_label)
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


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate CAW metrics by training crop length.'
    )
    parser.add_argument(
        '--outputs_root',
        default=os.path.join('Catch-A-Waveform', 'outputs', '_exp_crops'),
        help='Root directory containing run folders.',
    )
    parser.add_argument(
        '--out_dir',
        default=os.path.join('Catch-A-Waveform', 'outputs', 'metrics_len'),
        help='Directory to write CSV summaries.',
    )
    parser.add_argument('--normalize', action='store_true', help='Normalize before metrics.')
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--nperseg', type=int, default=2048, help='nperseg for coherence.')

    args = parser.parse_args()

    outputs_root = os.path.abspath(args.outputs_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    by_length_scale = {}
    curve_acc = {}

    for entry in sorted(os.listdir(outputs_root)):
        run_dir = os.path.join(outputs_root, entry)
        if not os.path.isdir(run_dir):
            continue
        length_sec = _parse_length(entry)
        if length_sec is None:
            continue

        real_map = { _key_for(p): p for p in glob.glob(os.path.join(run_dir, 'real@*Hz.wav')) }
        fake_map = { _key_for(p): p for p in glob.glob(os.path.join(run_dir, 'fake@*Hz.wav')) }
        rec_map = { _key_for(p): p for p in glob.glob(os.path.join(run_dir, 'reconstructed@*Hz.wav')) }

        keys = sorted(set(real_map) & set(fake_map), key=_scale_key)
        if not keys:
            continue

        for key in keys:
            real_path = real_map[key]
            fake_path = fake_map[key]
            ref, est_fake, ref_sr = _load_and_align(real_path, fake_path, args.normalize)

            fake_snr = calc_snr(est_fake, ref)
            fake_lsd = calc_lsd(est_fake, ref)
            coh_freqs, coh = signal.coherence(ref, est_fake, fs=ref_sr, nperseg=args.nperseg)
            fake_coh_mean = float(np.mean(coh)) if coh.size else float('nan')

            rec_snr = float('nan')
            rec_lsd = float('nan')
            rec_coh_mean = float('nan')
            if key in rec_map:
                ref_rec, est_rec, _ = _load_and_align(real_path, rec_map[key], args.normalize)
                rec_snr = calc_snr(est_rec, ref_rec)
                rec_lsd = calc_lsd(est_rec, ref_rec)
                coh_freqs, coh = signal.coherence(ref_rec, est_rec, fs=ref_sr, nperseg=args.nperseg)
                rec_coh_mean = float(np.mean(coh)) if coh.size else float('nan')

            agg_key = (length_sec, key)
            entry_stats = by_length_scale.setdefault(agg_key, {'fake': [], 'rec': []})
            entry_stats['fake'].append((fake_snr, fake_lsd, fake_coh_mean))
            if key in rec_map:
                entry_stats['rec'].append((rec_snr, rec_lsd, rec_coh_mean))

            lsd_freqs, lsd_curve, psd_freqs, psd_ref, psd_est, coh_freqs, coh = _compute_curves(
                ref, est_fake, ref_sr, args.n_fft, args.hop_length, args.nperseg
            )
            _accumulate_curves(
                curve_acc,
                length_sec,
                key,
                'fake',
                lsd_freqs,
                lsd_curve,
                psd_freqs,
                psd_ref,
                psd_est,
                coh_freqs,
                coh,
            )
            if key in rec_map:
                lsd_freqs, lsd_curve, psd_freqs, psd_ref, psd_est, coh_freqs, coh = _compute_curves(
                    ref_rec, est_rec, ref_sr, args.n_fft, args.hop_length, args.nperseg
                )
                _accumulate_curves(
                    curve_acc,
                    length_sec,
                    key,
                    'reconstructed',
                    lsd_freqs,
                    lsd_curve,
                    psd_freqs,
                    psd_ref,
                    psd_est,
                    coh_freqs,
                    coh,
                )

    _write_length_plots(out_dir, curve_acc)
    _plot_compare_lengths(out_dir, curve_acc, lengths=[1.0, 5.0, 20.0], kind='fake')
    _plot_metric_summaries(out_dir, by_length_scale)
    print('Wrote plots to:', os.path.join(out_dir, 'plots_by_length'))
    print('Wrote plots to:', os.path.join(out_dir, 'plots_by_length_mean'))
    print('Wrote plots to:', os.path.join(out_dir, 'plots_metric_summary'))
    print('Wrote plots to:', os.path.join(out_dir, 'plots_by_length_compare'))


def _accumulate_curves(curve_acc, length_sec, key, kind,
                       lsd_freqs, lsd_curve, psd_freqs, psd_ref, psd_est, coh_freqs, coh):
    acc_key = (length_sec, key, kind)
    entry = curve_acc.get(acc_key)
    if entry is None:
        entry = {
            'lsd_freqs': lsd_freqs,
            'psd_freqs': psd_freqs,
            'coh_freqs': coh_freqs,
            'lsd': [],
            'psd_ref': [],
            'psd_est': [],
            'coh': [],
        }
        curve_acc[acc_key] = entry

    entry['lsd'].append(_interp_to_grid(entry['lsd_freqs'], lsd_freqs, lsd_curve))
    entry['psd_ref'].append(_interp_to_grid(entry['psd_freqs'], psd_freqs, psd_ref))
    entry['psd_est'].append(_interp_to_grid(entry['psd_freqs'], psd_freqs, psd_est))
    entry['coh'].append(_interp_to_grid(entry['coh_freqs'], coh_freqs, coh))


def _write_length_plots(out_dir, curve_acc):
    plots_dir = os.path.join(out_dir, 'plots_by_length')
    plots_mean_dir = os.path.join(out_dir, 'plots_by_length_mean')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plots_mean_dir, exist_ok=True)

    for (length_sec, key, kind), entry in curve_acc.items():
        lsd_mean = np.mean(np.vstack(entry['lsd']), axis=0)
        psd_ref_mean = np.mean(np.vstack(entry['psd_ref']), axis=0)
        psd_est_mean = np.mean(np.vstack(entry['psd_est']), axis=0)
        coh_mean = np.mean(np.vstack(entry['coh']), axis=0)

        suffix = 'L%.2f__%s__%s__curves.png' % (length_sec, key, kind)
        out_path = os.path.join(plots_dir, suffix)
        _plot_curves(
            out_path,
            entry['lsd_freqs'],
            lsd_mean,
            entry['psd_freqs'],
            psd_ref_mean,
            psd_est_mean,
            entry['coh_freqs'],
            coh_mean,
            kind,
        )

        mean_suffix = 'L%.2f__%s__%s__mean_curves.png' % (length_sec, key, kind)
        mean_out_path = os.path.join(plots_mean_dir, mean_suffix)
        _plot_curves(
            mean_out_path,
            entry['lsd_freqs'],
            lsd_mean,
            entry['psd_freqs'],
            psd_ref_mean,
            psd_est_mean,
            entry['coh_freqs'],
            coh_mean,
            kind,
        )


def _plot_compare_lengths(out_dir, curve_acc, lengths, kind):
    import matplotlib.pyplot as plt

    colors = ['green', 'purple', 'orange']
    plots_dir = os.path.join(out_dir, 'plots_by_length_compare')
    os.makedirs(plots_dir, exist_ok=True)

    selected = []
    for length_sec in lengths:
        keys = [k for (L, k, t) in curve_acc.keys() if L == length_sec and t == kind]
        if not keys:
            continue
        max_key = max(keys, key=_scale_key)
        entry = curve_acc[(length_sec, max_key, kind)]
        lsd_mean = np.mean(np.vstack(entry['lsd']), axis=0)
        coh_mean = np.mean(np.vstack(entry['coh']), axis=0)
        selected.append({
            'length': length_sec,
            'key': max_key,
            'lsd_freqs': entry['lsd_freqs'],
            'lsd': lsd_mean,
            'coh_freqs': entry['coh_freqs'],
            'coh': coh_mean,
        })

    if not selected:
        return

    base_lsd_freqs = selected[0]['lsd_freqs']
    base_coh_freqs = selected[0]['coh_freqs']

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    for idx, entry in enumerate(selected):
        color = colors[idx % len(colors)]
        label = 'L%.2fs (%s)' % (entry['length'], entry['key'])
        lsd_vals = _interp_to_grid(base_lsd_freqs, entry['lsd_freqs'], entry['lsd'])
        coh_vals = _interp_to_grid(base_coh_freqs, entry['coh_freqs'], entry['coh'])
        axes[0].plot(base_lsd_freqs, lsd_vals, color=color, label=label)
        axes[1].plot(base_coh_freqs, coh_vals, color=color, label=label)

    axes[0].set_title('LSD vs Frequency (max downsampling scale)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('LSD')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title('Spectral Coherence vs Frequency (max downsampling scale)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Coherence')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(plots_dir, 'lsd_coherence_compare_max_scale.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_metric_summaries(out_dir, by_length_scale):
    import matplotlib.pyplot as plt

    metrics = [
        ('snr_db', 0),
        ('lsd', 1),
        ('coherence_mean', 2),
    ]
    kinds = ['fake', 'rec']

    agg = {kind: {} for kind in kinds}
    for (length_sec, _key), bucket in by_length_scale.items():
        for kind in kinds:
            values = bucket[kind]
            if not values:
                continue
            agg[kind].setdefault(length_sec, []).extend(values)

    plots_dir = os.path.join(out_dir, 'plots_metric_summary')
    os.makedirs(plots_dir, exist_ok=True)

    for metric_name, idx in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for kind, label in [('fake', 'Fake'), ('rec', 'Reconstructed')]:
            lengths = sorted(agg[kind].keys())
            if not lengths:
                continue
            means = []
            stds = []
            for L in lengths:
                arr = np.array(agg[kind][L], dtype=float)
                vals = arr[:, idx]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.errorbar(lengths, means, yerr=stds, marker='o', capsize=3, label=label)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel('Input length (s)')
        ax.set_ylabel(metric_name)
        ax.set_title('%s vs input length' % metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, 'summary_%s.png' % metric_name), dpi=150)
        plt.close(fig)




if __name__ == '__main__':
    main()
