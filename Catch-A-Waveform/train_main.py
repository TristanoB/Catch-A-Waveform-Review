from utils.utils import *
from utils.utils import save_debug_plots
import numpy as np
import glob
from params import Params
from training import train
from utils.plotters import *
import time
from datetime import datetime
import argparse
from generating import AudioGenerator
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", help="GPU id to use", default=None, type=int)
    parser.add_argument("--input_file", help="Path to input file", default=None)
    parser.add_argument(
        "--start_time", help="Skip beginning, in [sec]", default=None, type=float
    )
    parser.add_argument(
        "--max_length", help="Max length of signal, in [sec]", default=None, type=float
    )
    parser.add_argument(
        "--segments_to_train",
        default=None,
        type=float,
        nargs="+",
        help="Train on several segments of input signal, please provide segements in the form: start1, end1, start2, end2,... in [sec]",
    )
    parser.add_argument(
        "--init_sample_rate",
        help="Resample input to a given sample rate",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of training epochs in each scale",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--num_layers", help="Number of layers in each model", default=None, type=int
    )
    parser.add_argument(
        "--device",
        help="Device override: cpu | mps | cuda | cuda:0 ...",
        default=None,
        type=str,
    )
    parser.add_argument("--speech", default=None, action="store_true")
    parser.add_argument(
        "--run_mode",
        default=None,
        type=str,
        choices=["normal", "inpainting", "denoising"],
    )
    parser.add_argument(
        "--model_type", default=None, type=str, choices=["gan", "diffusion"]
    )
    parser.add_argument("--diffusion_steps", default=None, type=int)
    parser.add_argument("--diffusion_beta_start", default=None, type=float)
    parser.add_argument("--diffusion_beta_end", default=None, type=float)
    parser.add_argument(
        "--diffusion_beta_schedule",
        default=None,
        type=str,
        choices=["linear", "cosine"],
    )
    parser.add_argument("--diffusion_clip_denoised", default=None, action="store_true")
    parser.add_argument(
        "--fs_list",
        nargs="+",
        type=int,
        default=None,
        help="Override list of sample rates (Hz) for multiscale pyramid; e.g., --fs_list 4000 for single scale",
    )
    parser.add_argument(
        "--no_energy_gate",
        action="store_true",
        help="Disable energy-based skipping of the first scale",
    )
    parser.add_argument(
        "--hidden_channels_init",
        type=int,
        default=None,
        help="Base number of channels in the first UNet/Generator layer",
    )
    parser.add_argument(
        "--filter_size",
        type=int,
        default=None,
        help="Convolution kernel size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--sampling_noise_scale",
        type=float,
        default=None,
        help="Scale of noise during diffusion sampling (0 -> deterministic)",
    )
    parser.add_argument(
        "--deterministic_sampling",
        action="store_true",
        help="Force zero noise during sampling (equivalent to sampling_noise_scale=0)",
    )
    parser.add_argument(
        "--use_generated_condition",
        action="store_true",
        help="During diffusion training, condition on generated upper-scale signals (default is teacher forcing on real coarse)",
    )
    parser.add_argument(
        "--save_prev_cond",
        action="store_true",
        help="Save the conditioning signal (per scale) used during diffusion training for debugging",
    )
    parser.add_argument(
        "--save_noisy_debug",
        action="store_true",
        help="Save a maximally noised forward sample per scale for debugging (diffusion only)",
    )
    parser.add_argument(
        "--alpha1",
        help="Time-domain reconstruction loss weight (useful for diffusion)",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--alpha2",
        help="Frequency-domain reconstruction loss weight",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--cond_refresh_every",
        help="Refresh rate (epochs) for cached conditioning in diffusion training",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--inpainting_indices",
        default=None,
        nargs="+",
        type=int,
        help="Start and end indices of hole (for inpainting)",
    )
    parser.add_argument(
        "--plot_losses",
        help="Save and plot GAN losses",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--plot_signals", help="Plot signals", default=None, action="store_true"
    )

params_override = parser.parse_args()

startTime = time.time()
params = Params()
params = override_params(params, params_override)
# Optional CLI switches not directly mapped to Params fields
if getattr(params_override, "no_energy_gate", False):
    params.set_first_scale_by_energy = False
if getattr(params_override, "deterministic_sampling", False):
    params.deterministic_sampling = True

# Optional sampling noise scale override
if params_override.sampling_noise_scale is not None:
    params.sampling_noise_scale = params_override.sampling_noise_scale

# Conditioning strategy override
if getattr(params_override, "use_generated_condition", False):
    params.teacher_force_condition = False
if getattr(params_override, "save_prev_cond", False):
    params.save_prev_cond = True
if getattr(params_override, "save_noisy_debug", False):
    params.save_noisy_debug = True
# move CLI device override into device_str
if isinstance(params.device, str) and params.device != "":
    params.device_str = params.device
    params.device = torch.device("cpu")
params.set_device()

if len(params.inpainting_indices) % 2 != 0:
    raise Exception("Provide START and END indices of each hole!")

if params.device.type == "cuda":
    torch.cuda.set_device(params.gpu_num)
    params.device = torch.device(f"cuda:{params.gpu_num}")

if params.manual_random_seed != -1:
    random.seed(params.manual_random_seed)
    torch.manual_seed(params.manual_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Get input signal
samples = get_input_signal(params)
# Normalize array type for torch (avoids numpy 2.x dtype inference issues)
samples = np.asarray(samples, dtype=np.float32)
# set scales
params.fs_list = [f for f in params.fs_list if f <= params.Fs]
if params.fs_list[-1] != params.Fs:
    params.fs_list.append(params.Fs)
params.scales = [params.Fs / f for f in params.fs_list]

print("Working on file: %s" % params.input_file)

# Create a random hole for inpainting
if params.run_mode == "inpainting":
    samples_orig = samples.copy()
    params.inpainting_indices = list(
        zip(params.inpainting_indices[0::2], params.inpainting_indices[1::2])
    )
    for hole_idx in params.inpainting_indices:
        samples[hole_idx[0] : hole_idx[1]] = 0

# Set params by run_node and signal type
params.scheduler_milestones = [int(params.num_epochs * 2 / 3)]
if params.speech:
    params.alpha1 = 10
    params.alpha2 = 0
    params.add_cond_noise = False
else:
    if params.run_mode == "normal":
        params.alpha1 = 0
        params.alpha2 = 1e-4
        params.add_cond_noise = True
    else:
        params.alpha1 = 10
        params.alpha2 = 0
        if params.run_mode == "inpainting":
            params.add_cond_noise = True
        else:
            params.add_cond_noise = False
params.dilation_factors = [2**i for i in range(params.num_layers)]
# For diffusion models, keep a reconstruction pull by default unless user overrode.
if params.model_type == "diffusion" and params_override.alpha1 is None:
    params.alpha1 = 10
if params.model_type == "diffusion" and params_override.alpha2 is None:
    params.alpha2 = 0

# Create output folder
if not os.path.exists("outputs"):
    os.mkdir("outputs")

base_out = params.output_folder
suffix = 0
while os.path.exists(params.output_folder):
    suffix += 1
    params.output_folder = f"{base_out}_{suffix}"
# Allow nested paths when input_file includes subfolders (e.g. _exp_crops/...).
os.makedirs(params.output_folder, exist_ok=False)
print("Writing results to %s\n" % params.output_folder)

if params.run_mode == "inpainting":
    write_signal(
        os.path.join(params.output_folder, "Original.wav"), samples_orig, params.Fs
    )

# samples = samples.reshape((1, -1))

# Create input signal for each scale (avoid torch<->numpy bridge)
signals_list, fs_list = create_input_signals(
    params, torch.tensor(samples.tolist(), dtype=torch.float32), params.Fs
)
if len(signals_list) == 0:
    params.set_first_scale_by_energy = False
    params.scales = params.scales[2:]  # Manually start from 500
    signals_list, fs_list = create_input_signals(
        params, torch.tensor(samples.tolist(), dtype=torch.float32), params.Fs
    )
params.scales = [params.Fs / f for f in fs_list]
params.fs_list = fs_list
params.inputs_lengths = [len(s) for s in signals_list]

# Write parameters of run to a text file
with open(os.path.join(params.output_folder, "log.txt"), "w") as f:
    f.write("".join(["%s = %s\n" % (k, v) for k, v in params.__dict__.items()]))

if params.run_mode == "inpainting":
    # create masks for inpainting
    params.masks = []
    for scale, real_signal in zip(params.scales, signals_list):
        idcs = np.array(range(len(real_signal)))
        total_mask = np.ones(len(real_signal), dtype=bool)
        for hole_idx in params.inpainting_indices:
            cur_hole_start_idx = int(hole_idx[0] / scale)
            cur_hole_end_idx = int(hole_idx[1] / scale)
            current_mask = np.logical_or(
                idcs < cur_hole_start_idx, idcs >= cur_hole_end_idx
            )
            total_mask = np.logical_and(current_mask, total_mask)
        params.masks.append(torch.Tensor(total_mask).bool().to(params.device))

print("Running on " + str(params.device))

# Start training
(
    output_signals,
    loss_vectors,
    generators_list,
    noise_amp_list,
    energy_list,
    reconstruction_noise_list,
) = train(params, signals_list)

# Save reconstruction noise list
torch.save(
    reconstruction_noise_list,
    os.path.join(params.output_folder, "reconstruction_noise_list.pt"),
)

with open(os.path.join(params.output_folder, "log.txt"), "a") as f:
    f.write("\nTotal Runtime is: %d minutes" % ((time.time() - startTime) / 60))
    f.write("\n Finished running in : %s" % datetime.fromtimestamp(time.time()))

##############
# Generating #
##############
audio_generator = AudioGenerator(
    params,
    generators_list,
    noise_amp_list,
    reconstruction_noise_list=reconstruction_noise_list,
)
if not params.run_mode == "inpainting":
    audio_generator.generate()
    audio_generator.reconstruct()
else:
    audio_generator.inpaint()

#################
# Plotting Area #
#################
# Plot Signals
if params.plot_signals:
    os.mkdir(os.path.join(params.output_folder, "figures"))
    for real_signal, outputs, fs in zip(signals_list, output_signals, params.fs_list):
        output_file(os.path.join(params.output_folder, "figures", "%dHz" % fs))
        args = [real_signal, outputs["fake_signal"]]
        labels = ["Real Signal", "Fake Signal"]
        if "reconstructed_signal" in outputs:
            args.insert(1, outputs["reconstructed_signal"])
            labels.insert(1, "Reconstructed Signal")
        plot_signal_time_freq(*args, Fs=fs, labels=labels)
# Plot losses
if params.plot_losses:
    if not os.path.exists(os.path.join(params.output_folder, "figures")):
        os.mkdir(os.path.join(params.output_folder, "figures"))
    plot_losses(params, loss_vectors)

# Save spectrogram and frequency metrics for final scale (best-effort)
try:
    final_fs = params.fs_list[-1]
    real_path = os.path.join(params.output_folder, f"real@{final_fs}Hz.wav")
    fake_path = os.path.join(params.output_folder, f"fake@{final_fs}Hz.wav")
    if os.path.exists(real_path) and os.path.exists(fake_path):
        plots_dir = os.path.join(params.output_folder, "figures")
        save_debug_plots(real_path, fake_path, plots_dir, sr=final_fs)
except Exception as exc:
    print(f"[post-run plots] skipped: {exc}")
