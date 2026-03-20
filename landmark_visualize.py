import numpy as np
import cv2
import os

# Load the video
video_path = "dataset/M8/angry/level_1/027.mp4"  # Replace accordingly
landmark_path = "facial_landmarks/M8_angry_level_1_027.npy"

# Check files
assert os.path.exists(video_path), "❌ Video file not found!"
assert os.path.exists(landmark_path), "❌ Landmark file not found!"

landmarks = np.load(landmark_path)
print("✅ Loaded landmarks with shape:", landmarks.shape)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

cv2.namedWindow("Landmarks on Video", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(landmarks):
        break

    frame_landmarks = landmarks[frame_idx]
    for lm in frame_landmarks:
        x = int(lm[0] * frame.shape[1])  # width
        y = int(lm[1] * frame.shape[0])  # height
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Landmarks on Video", frame)
    key = cv2.waitKey(30)  # or use 0 for testing
    if key == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
