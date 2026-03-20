import os
import numpy as np
from tqdm import tqdm

MFCC_DIR = "audio_mfcc"
LANDMARK_DIR = "facial_landmarks"
OUTPUT_DIR = "training_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 20
OUT_LEN = 5

# MediaPipe inner lip indices
INNER_LIP_IDX = [
    78,191,80,81,82,
    13,312,311,310,415,
    308,324,318,402,317,
    14,87,178,88,95
]

mfcc_files = sorted([f for f in os.listdir(MFCC_DIR) if f.endswith(".npy")])

X_all = []
Y_centered_all = []
Centers_all = []

missing_landmarks = 0
too_short = 0


for mfcc_file in tqdm(mfcc_files, desc="Creating training sequences"):

    mfcc_path = os.path.join(MFCC_DIR, mfcc_file)
    lm_path = os.path.join(LANDMARK_DIR, mfcc_file)  # same filename

    if not os.path.exists(lm_path):
        missing_landmarks += 1
        continue

    mfcc = np.load(mfcc_path)          # (T_mfcc, 13)
    landmarks = np.load(lm_path)       # (T_land, 468, 3)

    # select lip points
    lips = landmarks[:, INNER_LIP_IDX, :2]   # (T_land, 20, 2)

    T_mfcc = mfcc.shape[0]
    T_land = lips.shape[0]

    min_len = min(T_mfcc, T_land)

    if min_len < WINDOW_SIZE + OUT_LEN:
        too_short += 1
        continue

    mfcc = mfcc[:min_len]
    lips = lips[:min_len]

    # sliding window
    for i in range(min_len - (WINDOW_SIZE + OUT_LEN) + 1):

        X_seq = mfcc[i : i + WINDOW_SIZE]

        Y_seq_frames = lips[
            i + WINDOW_SIZE : i + WINDOW_SIZE + OUT_LEN
        ]

        # compute lip center
        centers = Y_seq_frames.mean(axis=1)   # (OUT_LEN,2)

        # center lip coordinates
        Y_centered = Y_seq_frames - centers[:, None, :]

        # flatten lips
        Y_centered_flat = Y_centered.reshape(OUT_LEN, -1)

        X_all.append(X_seq)
        Y_centered_all.append(Y_centered_flat)
        Centers_all.append(centers)


# convert to numpy
X_all = np.array(X_all, dtype=np.float32)
Y_centered_all = np.array(Y_centered_all, dtype=np.float32)
Centers_all = np.array(Centers_all, dtype=np.float32)


# save
np.save(os.path.join(OUTPUT_DIR, "X_seq.npy"), X_all)
np.save(os.path.join(OUTPUT_DIR, "Y_centered_seq.npy"), Y_centered_all)
np.save(os.path.join(OUTPUT_DIR, "Centers_seq.npy"), Centers_all)


print("\nDataset creation complete")
print("X:", X_all.shape)
print("Y_centered:", Y_centered_all.shape)
print("Centers:", Centers_all.shape)

print("\nStats")
print("Missing landmark files:", missing_landmarks)
print("Too short videos:", too_short)
print("Total sequences:", len(X_all))