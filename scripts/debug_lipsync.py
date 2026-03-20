import cv2
import numpy as np
import mediapipe as mp
import subprocess
import librosa
import torch
import matplotlib.pyplot as plt

from MFCC import mfcc_30fps
from Model import CNN_TCN_Seq


# ===============================
# CONFIG
# ===============================

VIDEO_PATH = "dataset/M1/neutral/level_1/007.mp4"
MODEL_PATH = "model.pth"
TEMP_AUDIO = "temp_audio.wav"

WINDOW_SIZE = 20
OUT_LEN = 5

# ===============================
# Lip landmark indices
# ===============================

INNER_LIP_IDX = [
    78,191,80,81,82,
    13,312,311,310,415,
    308,324,318,402,317,
    14,87,178,88,95
]

UPPER_MID = 5
LOWER_MID = 15


# ===============================
# Step 1: Extract Audio
# ===============================

cmd = [
    "ffmpeg",
    "-loglevel", "error",
    "-y",
    "-i", VIDEO_PATH,
    "-ac", "1",
    "-ar", "16000",
    TEMP_AUDIO
]

subprocess.run(cmd)

signal, sr = librosa.load(TEMP_AUDIO, sr=16000)

mfcc = mfcc_30fps(signal, sr)


# ===============================
# Step 2: Extract GT landmarks
# ===============================

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(VIDEO_PATH)

gt_lips = []

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            lm = results.multi_face_landmarks[0]

            points = []

            for idx in INNER_LIP_IDX:

                x = lm.landmark[idx].x
                y = lm.landmark[idx].y

                points.append([x, y])

            gt_lips.append(points)

cap.release()

gt_lips = np.array(gt_lips)


# ===============================
# Step 3: Align lengths
# ===============================

T = min(len(gt_lips), len(mfcc))

gt_lips = gt_lips[:T]
mfcc = mfcc[:T]


# ===============================
# Step 4: Load model
# ===============================

model = CNN_TCN_Seq()

state = torch.load(MODEL_PATH, map_location="cpu")

model.load_state_dict(state)

model.eval()


# ===============================
# Step 5: Predict lips
# ===============================

pred_lips = []

with torch.no_grad():

    for i in range(T - WINDOW_SIZE - OUT_LEN):

        x = mfcc[i:i+WINDOW_SIZE]

        x = torch.tensor(x).unsqueeze(0).float()

        y = model(x)

        y = y.squeeze().numpy()

        y = y.reshape(OUT_LEN,20,2)

        pred_lips.extend(y)

pred_lips = np.array(pred_lips)


# ===============================
# Step 6: Match GT length
# ===============================

min_len = min(len(gt_lips), len(pred_lips))

gt_lips = gt_lips[:min_len]
pred_lips = pred_lips[:min_len]


# ===============================
# Step 7: Lip opening
# ===============================

gt_open = []
pred_open = []

for i in range(min_len):

    gt_dist = np.linalg.norm(
        gt_lips[i][UPPER_MID] - gt_lips[i][LOWER_MID]
    )

    pr_dist = np.linalg.norm(
        pred_lips[i][UPPER_MID] - pred_lips[i][LOWER_MID]
    )

    gt_open.append(gt_dist)
    pred_open.append(pr_dist)


# ===============================
# Step 8: Plot
# ===============================

plt.figure(figsize=(10,4))

plt.plot(gt_open, label="Ground Truth")
plt.plot(pred_open, label="Prediction")

plt.title("Lip Opening Over Time")
plt.xlabel("Frame")
plt.ylabel("Lip Distance")

plt.legend()
plt.show()

corr = np.corrcoef(gt_open, pred_open)[0,1]

print("Lip motion correlation:", corr)