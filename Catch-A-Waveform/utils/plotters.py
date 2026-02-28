from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Category20
from bokeh.layouts import column
import numpy as np
import numpy.fft as fft
import os
import torch


FIG_WIDTH = 1200
CLIP_LO = 1  # percentile low for outlier clipping in plots
CLIP_HI = 99  # percentile high for outlier clipping in plots


def _clip_series(arr, q_low=CLIP_LO, q_high=CLIP_HI):
    """Clip extreme outliers for visualization only (does not modify training logs)."""
    lo = np.nanpercentile(arr, q_low)
    hi = np.nanpercentile(arr, q_high)
    return np.clip(arr, lo, hi)


def _clip_named(name, arr):
    # Tighten clipping for diffusion eps/rec losses to better see trends when spikes occur.
    if name in ("v_eps_loss", "v_rec_loss"):
        return _clip_series(arr, q_low=5, q_high=95)
    return _clip_series(arr)


def plot(x, y=None, labels=None):
    p = figure(width=FIG_WIDTH)
    if y is None:
        y = x.copy()
        x = range(y.shape[0])
    if len(y.shape) > 1:
        for i in range(y.shape[1]):
            if labels is None:
                p.line(x, y[:, i], color=Category20[20][i % 20])
            else:
                p.line(x, y[:, i], color=Category20[20][i % 20], legend_label=labels[i])
            # p.scatter(x, y[:, i], color=Category20[20][i % 20])
        p.legend.click_policy = 'hide'
    else:
        p.line(x, y, color=Category20[20][0])
        # p.scatter(x, y, color=Category20[20][0])

    show(p)


def plot_losses(params, loss_vectors):
    # Plot losses in each scale
    figures_dir = os.path.join(params.output_folder, "..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    run_id = os.path.basename(params.output_folder.rstrip("/"))
    output_file(os.path.join(figures_dir, f"{run_id}_losses.html"))
    p_vec: list = []
    eps_curves = []
    for losses, fs in zip(loss_vectors, params.fs_list):
        figs = []
        color_idx = 0

        # Adversarial losses figure (if any)
        has_adv = any(k in losses for k in ("v_err_real", "v_err_fake", "v_gp"))
        if has_adv:
            padv = figure(title='Losses @ %dHz (adv)' % fs, width=FIG_WIDTH)
            padv.title.align = "center"
            padv.xaxis.axis_label = 'Epoch#'
            if "v_err_real" in losses:
                series = _clip_named("v_err_real", -losses["v_err_real"])
                padv.line(range(params.num_epochs), series, legend_label="D(real)", color=Category20[20][color_idx]); color_idx += 1
            if "v_err_fake" in losses:
                series = _clip_named("v_err_fake", losses["v_err_fake"])
                padv.line(range(params.num_epochs), series, legend_label="D(fake)", color=Category20[20][color_idx]); color_idx += 1
            if "v_gp" in losses:
                series = _clip_named("v_gp", losses["v_gp"])
                padv.line(range(params.num_epochs), series, legend_label="Gradient Penalty", color=Category20[20][color_idx]); color_idx += 1
            padv.legend.click_policy = "hide"
            figs.append(padv)

        # Eps/Rec figure (separate axis and scale)
        has_eps = "v_eps_loss" in losses
        has_rec = "v_rec_loss" in losses
        if has_eps or has_rec:
            peps = figure(title='Losses @ %dHz (eps/rec)' % fs, width=FIG_WIDTH)
            peps.title.align = "center"
            peps.xaxis.axis_label = 'Epoch#'
            if has_eps:
                series = _clip_named("v_eps_loss", losses["v_eps_loss"])
                peps.line(range(params.num_epochs), series, legend_label="Eps Loss", color=Category20[20][color_idx]); color_idx += 1
                eps_curves.append((fs, series))
            if has_rec:
                series = _clip_named("v_rec_loss", losses["v_rec_loss"])
                peps.line(range(params.num_epochs), series, legend_label="Rec. Loss", color=Category20[20][color_idx]); color_idx += 1
            peps.legend.click_policy = "hide"
            figs.append(peps)

        if figs:
            p_vec.append(column(figs))
    # Combined eps loss plot (all scales), appended once
    if eps_curves:
        p_all = figure(title="Eps Loss (all scales)", width=FIG_WIDTH)
        p_all.xaxis.axis_label = "Epoch#"
        for idx, (fs, series) in enumerate(eps_curves):
            p_all.line(
                range(params.num_epochs),
                series,
                legend_label=f"Eps @ {fs}Hz",
                color=Category20[20][idx % 20],
            )
        p_all.legend.click_policy = "hide"
        p_vec.append(p_all)
    if p_vec:
        show(column(p_vec))


def plot_signal_time_freq(*args, Fs=16000, labels=None):
    if np.isscalar(Fs):
        Fs = np.ones(len(args)) * Fs
    p_time = figure(title="Signal in Time", width=FIG_WIDTH)
    p_freq = figure(title="Signal in Freq", width=FIG_WIDTH)
    for idx, signal in enumerate(args):
        if torch.is_tensor(signal):
            signal = np.array(signal.tolist())
        if signal.ndim > 1:
            signal = np.squeeze(signal)
        if signal.ndim > 1:
            n_signals = signal.shape[0]
            for idx_2 in range(n_signals):
                cur_signal = signal[idx_2, :]
                N = len(cur_signal)
                t_vec = [i / Fs[idx] for i in range(N)]
                fft_size = int(2 ** np.ceil(np.log2(len(cur_signal) / 2)))
                freq_grid = [f / fft_size * Fs[idx] / 2 for f in range(fft_size)]
                S = fft.rfft(cur_signal, (fft_size - 1) * 2)
                legned_str = 'sig' + str(idx_2) if labels is None else labels[idx_2]
                p_time.scatter(t_vec, cur_signal, color=Category20[20][idx_2 % 20], legend_label=legned_str)
                p_freq.scatter(freq_grid, 20 * np.log10(abs(S)), color=Category20[20][idx_2 % 20], legend_label=legned_str)
        else:
            N = len(signal)
            t_vec = [i / Fs[idx] for i in range(N)]
            fft_size = int(2 ** np.ceil(np.log2(len(signal) / 2)))
            freq_grid = [f / fft_size * Fs[idx] / 2 for f in range(fft_size)]
            S = fft.rfft(signal, (fft_size - 1) * 2)
            legned_str = 'sig'+str(idx) if labels is None else labels[idx]
            p_time.scatter(t_vec, signal, color=Category20[20][idx], legend_label=legned_str)
            p_freq.scatter(freq_grid, 20*np.log10(abs(S)), color=Category20[20][idx], legend_label=legned_str)
    p_time.legend.click_policy = 'hide'
    p_freq.legend.click_policy = 'hide'

    show(column([p_time, p_freq]))
