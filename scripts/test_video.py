import os
import numpy as np
import torch
from Model import CNN_TCN_Seq
from scipy.signal import savgol_filter
from tqdm import tqdm
import moviepy as mpy
import librosa
import cv2
from MFCC import mfcc_30fps

# ============================
# Config
# ============================A
VIDEO_PATH = "test_video.mp4"
MODEL_PATH = "model.pth"
OUTPUT_VIDEO = "test_video_O1.mp4"

WINDOW_SIZE = 30
OUT_LEN = 5
CANVAS_W, CANVAS_H = 640, 480
FPS = 30
SR = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Load Model
# ============================
model = CNN_TCN_Seq(mfcc_dim=13, lip_points=20, out_len=OUT_LEN)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ============================
# 1) Extract audio from video
# ============================
print("🎧 Extracting audio from video...")

video = mpy.VideoFileClip(VIDEO_PATH)
audio = video.audio

temp_audio = "temp_from_video.wav"
audio.write_audiofile(temp_audio, fps=SR)

# ============================
# 2) Audio → MFCC (30 FPS)
# ============================
y, sr = librosa.load(temp_audio, sr=SR)
mfcc = mfcc_30fps(y, sr)     # (T,13)

print("MFCC shape:", mfcc.shape)

# ============================
# 3) Sliding window
# ============================
X = []
for i in range(WINDOW_SIZE, len(mfcc)):
    X.append(mfcc[i-WINDOW_SIZE:i])

X = np.array(X)
print("Windowed MFCC:", X.shape)

# ============================
# 4) Model inference
# ============================
preds = []

with torch.no_grad():
    for seq in tqdm(X, desc="Predicting"):
        inp = torch.tensor(seq).float().unsqueeze(0).to(device)
        out = model(inp).cpu().numpy()[0]   # (5,40)
        preds.append(out[OUT_LEN//2])       # center frame

preds = np.array(preds)   # (T,40)

# ============================
# 5) Smooth predictions
# ============================
for i in range(40):
    preds[:,i] = savgol_filter(preds[:,i], 9, 2)

# ============================
# 6) Render blank canvas
# ============================
frames = []

for i in tqdm(range(len(preds)), desc="Rendering"):
    img = np.zeros((CANVAS_H, CANVAS_W,3), dtype=np.uint8)

    pts = preds[i].reshape(20,2)

    # center to canvas
    center = np.array([0.5,0.5])
    pts = pts + center

    pts = (pts * [CANVAS_W,CANVAS_H]).astype(int)

    for p in pts:
        cv2.circle(img, tuple(p), 3, (0,0,255), -1)

    for k in range(len(pts)-1):
        cv2.line(img, tuple(pts[k]), tuple(pts[k+1]), (0,0,255), 2)

    frames.append(img)

# ============================
# 7) Save video + audio
# ============================
clip = mpy.ImageSequenceClip(
    [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames],
    fps=FPS
)

clip = clip.with_audio(audio)
clip.write_videofile(
    OUTPUT_VIDEO,
    codec="libx264",
    audio_codec="aac"
)

print("✅ FINAL OUTPUT:", OUTPUT_VIDEO)
