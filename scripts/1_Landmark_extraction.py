import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

INPUT_VIDEO_DIR = "dataset"
OUTPUT_DIR = "facial_landmarks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Initialize MediaPipe FaceMesh
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # returns 478 landmarks (we cut to 468)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================================
# Traverse dataset structure
# Actor → Emotion → Level → Video
# =========================================================

actors = sorted(os.listdir(INPUT_VIDEO_DIR))

for actor in actors:

    actor_path = os.path.join(INPUT_VIDEO_DIR, actor)

    if not os.path.isdir(actor_path):
        continue

    emotions = sorted(os.listdir(actor_path))

    for emotion in emotions:

        emotion_path = os.path.join(actor_path, emotion)

        if not os.path.isdir(emotion_path):
            continue

        levels = sorted(os.listdir(emotion_path))

        for level in levels:

            level_path = os.path.join(emotion_path, level)

            if not os.path.isdir(level_path):
                continue

            video_files = [f for f in os.listdir(level_path) if f.endswith(".mp4")]

            for video_file in tqdm(video_files, desc=f"{actor}/{emotion}/{level}"):

                video_path = os.path.join(level_path, video_file)

                cap = cv2.VideoCapture(video_path)

                landmarks_list = []

                while cap.isOpened():

                    success, frame = cap.read()

                    if not success:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    results = face_mesh.process(frame_rgb)

                    if results.multi_face_landmarks:

                        face_landmarks = results.multi_face_landmarks[0]

                        lm_list = np.array(
                            [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
                            dtype=np.float32
                        )

                        # Remove iris landmarks (478 → 468)
                        if lm_list.shape[0] > 468:
                            lm_list = lm_list[:468]

                        frame_landmarks = lm_list

                    else:

                        # Handle frame where face not detected
                        if landmarks_list:
                            frame_landmarks = landmarks_list[-1].copy()
                        else:
                            frame_landmarks = np.zeros((468, 3), dtype=np.float32)

                    landmarks_list.append(frame_landmarks)

                cap.release()

                if len(landmarks_list) == 0:
                    continue

                landmarks_array = np.stack(landmarks_list, axis=0)

                # Output filename
                output_name = f"{actor}_{emotion}_{level}_{video_file.replace('.mp4','.npy')}"
                output_path = os.path.join(OUTPUT_DIR, output_name)

                np.save(output_path, landmarks_array)

                print(f"✅ Saved {output_name} | Shape: {landmarks_array.shape}")

print("\n🎉 Landmark extraction finished!")