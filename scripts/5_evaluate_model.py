import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import matplotlib.pyplot as plt
import moviepy as mpy
import librosa

from Model import CNN_TCN_Seq
from MFCC import mfcc_30fps


# =================================
# PATHS
# =================================

VIDEO_PATH = "test_hello.mp4"
MODEL_PATH = "model.pth"

TRAIN_LOSS_PATH = "train_losses.npy"
VAL_LOSS_PATH = "val_losses.npy"

OUTPUT_DIR = "video_evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR,"loss_curve.png")
METRIC_FILE = os.path.join(OUTPUT_DIR,"evaluation_metrics.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================
# MEDIAPIPE SETTINGS
# =================================

INNER_LIP_IDX = [
    78,191,80,81,82,
    13,312,311,310,415,
    308,324,318,402,317,
    14,87,178,88,95
]

mp_face = mp.solutions.face_mesh


# =================================
# LOAD MODEL
# =================================

model = CNN_TCN_Seq(
    mfcc_dim=13,
    lip_points=20,
    out_len=5,
    tcn_channels=[128,128,128,128]
)

model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Model loaded")


# =================================
# EXTRACT AUDIO
# =================================

audio_path = os.path.join(OUTPUT_DIR,"temp_audio.wav")

video = mpy.VideoFileClip(VIDEO_PATH)
video.audio.write_audiofile(audio_path,fps=16000)

print("Audio extracted")


# =================================
# MFCC EXTRACTION
# =================================

y, sr = librosa.load(audio_path, sr=16000)
mfcc = mfcc_30fps(y, sr)

print("MFCC shape:", mfcc.shape)


# =================================
# CREATE MFCC WINDOWS
# =================================

WINDOW_SIZE = 20

X = []

for i in range(len(mfcc) - WINDOW_SIZE - 5):
    X.append(mfcc[i:i+WINDOW_SIZE])

X = np.array(X)

print("Windowed MFCC:", X.shape)


# =================================
# MODEL PREDICTION
# =================================

X_t = torch.tensor(X,dtype=torch.float32).to(DEVICE)

pred_all = []

with torch.no_grad():

    for i in range(0,len(X_t),128):

        batch = X_t[i:i+128]
        pred = model(batch)

        pred_all.append(pred.cpu().numpy())

pred_all = np.concatenate(pred_all)

print("Predictions:", pred_all.shape)


# =================================
# EXTRACT GT LANDMARKS
# =================================

cap = cv2.VideoCapture(VIDEO_PATH)

gt_landmarks = []

with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True) as face_mesh:

    while True:

        ret,frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        result = face_mesh.process(frame_rgb)

        if result.multi_face_landmarks:

            lm = result.multi_face_landmarks[0]

            lips = []

            for idx in INNER_LIP_IDX:

                x = lm.landmark[idx].x
                y = lm.landmark[idx].y

                lips.append([x,y])

            lips = np.array(lips)

            # CENTER LANDMARKS (same as training)
            center = np.mean(lips,axis=0)
            lips_centered = lips - center

            gt_landmarks.append(lips_centered.reshape(-1))

gt_landmarks = np.array(gt_landmarks)

cap.release()

print("GT landmarks:", gt_landmarks.shape)


# =================================
# TEMPORAL ALIGNMENT
# =================================

valid_samples = min(len(pred_all), len(gt_landmarks) - WINDOW_SIZE - 5)

pred = pred_all[:valid_samples]
gt = gt_landmarks

print("Aligned samples:", valid_samples)


# =================================
# LME CALCULATION (CORRECT FUTURE ALIGNMENT)
# =================================

frame_lme = []

for f in range(5):

    errors = []

    for i in range(valid_samples):

        pred_pts = pred[i,f].reshape(20,2)
        gt_pts = gt[i + WINDOW_SIZE + f].reshape(20,2)

        dist = np.sqrt(np.sum((pred_pts - gt_pts)**2, axis=1))
        errors.append(np.mean(dist))

    frame_lme.append(np.mean(errors))


global_lme = np.mean(frame_lme)

print("\nFrame-wise LME:")

for i,e in enumerate(frame_lme):
    print(f"Frame {i+1}: {e:.6f}")

print("\nGlobal LME:", global_lme)


# =================================
# VELOCITY ERROR
# =================================

frame_vel = []

for f in range(4):

    errors = []

    for i in range(valid_samples):

        pred_v = pred[i,f+1] - pred[i,f]

        gt_v = (
            gt[i + WINDOW_SIZE + f + 1]
            - gt[i + WINDOW_SIZE + f]
        )

        vel_err = np.mean((pred_v - gt_v)**2)

        errors.append(vel_err)

    frame_vel.append(np.mean(errors))


global_vel = np.mean(frame_vel)

print("\nFrame-wise Velocity Error:")

for i,e in enumerate(frame_vel):
    print(f"{i+1}->{i+2}: {e:.8f}")

print("\nGlobal Velocity Error:", global_vel)


# =================================
# SAVE METRICS
# =================================

with open(METRIC_FILE,"w") as f:

    f.write("Evaluation Metrics\n\n")

    f.write("Frame-wise LME\n")
    for i,e in enumerate(frame_lme):
        f.write(f"Frame{i+1}: {e}\n")

    f.write(f"\nGlobal LME: {global_lme}\n\n")

    f.write("Frame-wise Velocity\n")
    for i,e in enumerate(frame_vel):
        f.write(f"{i+1}->{i+2}: {e}\n")

    f.write(f"\nGlobal Velocity Error: {global_vel}\n")

print("Metrics saved:", METRIC_FILE)


# =================================
# LOSS CURVE
# =================================

if os.path.exists(TRAIN_LOSS_PATH) and os.path.exists(VAL_LOSS_PATH):

    train_loss = np.load(TRAIN_LOSS_PATH)
    val_loss = np.load(VAL_LOSS_PATH)

    EPOCHS = len(train_loss)

    plt.figure(figsize=(8,5))

    plt.plot(range(1,EPOCHS+1),
             train_loss,
             marker='o',
             label='Train Loss')

    plt.plot(range(1,EPOCHS+1),
             val_loss,
             marker='s',
             label='Validation Loss')

    plt.title("Training and Validation Loss Curve")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE + Velocity)")

    plt.grid(True)
    plt.legend()

    plt.xticks(range(1,EPOCHS+1))

    plt.tight_layout()

    plt.savefig(LOSS_PLOT_PATH,dpi=300)

    plt.show()

    print("Loss curve saved:", LOSS_PLOT_PATH)