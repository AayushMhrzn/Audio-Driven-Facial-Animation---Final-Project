import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIG
# ==============================
LANDMARK_DIR = "facial_landmarks"

# Pick ONE file to inspect (change name if needed)
LANDMARK_FILE = "test_video.npy"

INNER_LIP_IDX = [
    78, 191, 80, 81, 82,
    13,
    312, 311, 310, 415,
    308, 324, 318, 402, 317,
    14,
    87, 178, 88, 95
]

# ==============================
# LOAD LANDMARK DATA
# ==============================
lm_path = os.path.join(LANDMARK_DIR, LANDMARK_FILE)

if not os.path.exists(lm_path):
    raise FileNotFoundError(f"❌ File not found: {lm_path}")

landmarks = np.load(lm_path)
print("Landmarks shape:", landmarks.shape)
# Expected: (T, 468, 3)

# ==============================
# PICK A FRAME TO VISUALIZE
# ==============================
frame_idx = 53  # change if video is short
frame_landmarks = landmarks[frame_idx]  # (468, 3)

# ==============================
# EXTRACT INNER LIP POINTS
# ==============================
lips = frame_landmarks[INNER_LIP_IDX, :2]  # (20, 2)

# ==============================
# VISUALIZE
# ==============================
plt.figure(figsize=(6, 6))

# Scatter plot
plt.scatter(lips[:, 0], -lips[:, 1], c="red")

# Annotate each point with index 0–19
for i, (x, y) in enumerate(lips):
    plt.text(x, -y, str(i), fontsize=12, color="blue")

plt.title("Inner Lip Topology (Indexed)")
plt.axis("equal")
plt.grid(True)
plt.show()
