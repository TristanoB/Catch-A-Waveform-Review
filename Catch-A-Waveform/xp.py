import librosa, soundfile as sf
import numpy as np
import torch
from utils.resize_right import resize


in_wav = "/Users/etienne/Desktop/SDI/Deep Learning/project/Catch-A-Waveform-Review/Catch-A-Waveform/outputs/matuidi_charo_6/real@4000Hz.wav"
out_wav = "/Users/etienne/Desktop/SDI/Deep Learning/project/Catch-A-Waveform-Review/Catch-A-Waveform/outputs/matuidi_charo_6/real@4000Hz_lanczos2_safe.wav"
target_sr, kernel = 8000, "lanczos2_safe"

y, sr = librosa.load(in_wav, sr=None, mono=True)
y = np.asarray(y, dtype=np.float32)
y_t = torch.from_numpy(y)
y_out = resize(y_t, out_shape=(round(len(y_t) * target_sr / sr),), interp_method=kernel)
y_out = y_out.detach().cpu().numpy().astype(np.float32)
sf.write(out_wav, y_out, target_sr)
print("saved:", out_wav)
