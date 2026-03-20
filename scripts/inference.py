import json
import torch
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import random

from MFCC import mfcc_30fps
from extract_lip_features import extract_features
from normalize_features import FeatureNormalizer
from blendshape_mapper import map_to_gltf_blendshapes
from Model import CNN_TCN_Seq


# ---------------- CONFIG ----------------
MODEL_PATH = "G:\\MAJOR PROJECT\\model.pth"
FEATURE_STATS_PATH = "feature_stats.json"

WINDOW_SIZE = 20
OUT_LEN = 5
FPS = 30

OPEN_BOOST = 1.2
SMOOTHING_WINDOW = 5
SILENCE_THRESHOLD = 0.02  # energy threshold
NEUTRAL_WIDE = 0.6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- LOAD MODEL ----------------
model = CNN_TCN_Seq(mfcc_dim=13, lip_points=20, out_len=OUT_LEN)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

with open(FEATURE_STATS_PATH, "r") as f:
    feature_stats = json.load(f)

normalizer = FeatureNormalizer()
normalizer.stats = feature_stats


# ---------------- SMOOTH FUNCTION ----------------
def smooth_signal(signal, window=3):
    smoothed = np.copy(signal)
    for i in range(len(signal)):
        start = max(0, i - window)
        end = min(len(signal), i + window + 1)
        smoothed[i] = np.mean(signal[start:end])
    return smoothed


# ---------------- MAIN FUNCTION ----------------
def generate_animation_from_audio(audio_bytes: bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        sr, signal = wav.read(tmp_path)

        if signal.ndim > 1:
            signal = signal[:, 0]

        # Compute simple energy for silence detection
        energy = np.abs(signal)
        energy = energy / np.max(energy)

        mfcc = mfcc_30fps(signal, sr)
        T = mfcc.shape[0]

        windows = []
        for i in range(T - WINDOW_SIZE + 1):
            windows.append(mfcc[i:i+WINDOW_SIZE])

        windows = np.array(windows)
        mfcc_tensor = torch.tensor(windows, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            preds = model(mfcc_tensor)

        preds = preds.cpu().numpy()
        N = preds.shape[0]

        final_len = T
        accumulator = np.zeros((final_len, 40))
        counter = np.zeros((final_len, 1))

        for i in range(N):
            for j in range(OUT_LEN):
                frame_idx = i + WINDOW_SIZE + j
                if frame_idx < final_len:
                    accumulator[frame_idx] += preds[i, j]
                    counter[frame_idx] += 1

        counter[counter == 0] = 1
        final_predictions = accumulator / counter


        animation_data = {
            "fps": FPS,
            "blendshapes": [
                "open", "wide", "narrow",
                "upper_up", "lower_down",
                "frown", "wink"
            ],
            "frames": []
        }

        blendshape_tracks = {k: [] for k in animation_data["blendshapes"]}

        # -------- Generate Blendshapes --------
        for frame_idx in range(final_len):

            lips = final_predictions[frame_idx].reshape(20, 2)
            features = extract_features(lips)
            norm_features = normalizer.normalize(features)
            blendshapes = map_to_gltf_blendshapes(norm_features)

            # Silence detection
            is_silent = energy[int(frame_idx * len(signal)/final_len)] < SILENCE_THRESHOLD

            for k in animation_data["blendshapes"]:

                if k == "wink":
                    value = 0.0

                else:
                    value = float(blendshapes.get(k, 0))

                if is_silent:
                    if k == "wide":
                        value = NEUTRAL_WIDE
                    else:
                        value = 0.0

                else:
                    if k == "wide":
                        value = value**1.5
                    elif k == "narrow":
                        value = value**0.5
                    if k == "upper_up":
                        value = 0.0
                    if k =="frown":
                        value = min(0.7, value * 1.2)
                    if k == "open":
                        if value < 0.19:
                            value = 0
                        else:
                            value = value ** 0.4 # enhances subtle speech motion
                    
                        value *= OPEN_BOOST
                        value = min(value, 1)

                blendshape_tracks[k].append(value)


        # -------- Smooth All Except Wink --------
        for k in blendshape_tracks:
            if k != "wink":
                blendshape_tracks[k] = smooth_signal(
                    np.array(blendshape_tracks[k]),
                    window=SMOOTHING_WINDOW
                )


        # -------- Add Wink Every 4–5 sec --------
        total_seconds = final_len / FPS
        current_time = 0

        while current_time < total_seconds:
            blink_time = current_time + random.uniform(3, 4)
            blink_frame = int(blink_time * FPS)

            if blink_frame + 6 < final_len:
                for i in range(6):
                    t = i / 6
                    blink_value = np.sin(t * np.pi)
                    blendshape_tracks["wink"][blink_frame + i] = blink_value

            current_time = blink_time


        # -------- Smooth Ending Transition --------
        transition_frames = 15

        for k in blendshape_tracks:
            start_val = blendshape_tracks[k][0]
            for i in range(transition_frames):
                idx = final_len - transition_frames + i
                t = i / transition_frames
                blendshape_tracks[k][idx] = (
                    (1 - t) * blendshape_tracks[k][idx] +
                    t * start_val
                )


        # -------- Build Final JSON --------
        for frame_idx in range(final_len):
            frame_data = {"frame": frame_idx + 1}
            for k in animation_data["blendshapes"]:
                frame_data[k] = float(blendshape_tracks[k][frame_idx])
            animation_data["frames"].append(frame_data)

        return animation_data

    finally:
        os.remove(tmp_path)