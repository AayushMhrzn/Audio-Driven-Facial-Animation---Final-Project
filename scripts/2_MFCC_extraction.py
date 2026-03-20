import os
import numpy as np
import pandas as pd
import subprocess
import librosa
from tqdm import tqdm

from MFCC import mfcc_30fps


# ===============================
# PATHS
# ===============================

VIDEO_ROOT = "dataset"
OUTPUT_DIR = "audio_mfcc"
TEMP_AUDIO = "temp_audio.wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = []

actors = sorted(os.listdir(VIDEO_ROOT))


# ===============================
# Traverse Dataset
# ===============================

for actor in actors:

    actor_path = os.path.join(VIDEO_ROOT, actor)

    if not os.path.isdir(actor_path):
        continue

    emotions = os.listdir(actor_path)

    for emotion in emotions:

        emotion_path = os.path.join(actor_path, emotion)

        if not os.path.isdir(emotion_path):
            continue

        levels = os.listdir(emotion_path)

        for level in levels:

            level_path = os.path.join(emotion_path, level)

            if not os.path.isdir(level_path):
                continue

            videos = os.listdir(level_path)

            for video_file in tqdm(videos, desc=f"{actor}/{emotion}/{level}"):

                if not video_file.lower().endswith(".mp4"):
                    continue

                video_path = os.path.join(level_path, video_file)

                try:

                    # ===============================
                    # Extract audio using FFmpeg
                    # ===============================

                    cmd = [
                        "ffmpeg",
                        "-loglevel", "error",
                        "-y",
                        "-i", video_path,
                        "-ac", "1",        # mono
                        "-ar", "16000",    # 16kHz
                        TEMP_AUDIO
                    ]

                    subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )


                    # ===============================
                    # Load audio
                    # ===============================

                    signal, sr = librosa.load(TEMP_AUDIO, sr=16000)


                    # ===============================
                    # Compute MFCC
                    # ===============================

                    mfcc_feat = mfcc_30fps(signal, sr)


                    # ===============================
                    # Save MFCC
                    # ===============================

                    base_name = f"{actor}_{emotion}_{level}_{video_file.split('.')[0]}.npy"

                    out_path = os.path.join(OUTPUT_DIR, base_name)

                    np.save(out_path, mfcc_feat)


                    # ===============================
                    # Metadata
                    # ===============================

                    metadata.append({
                        "file": base_name,
                        "actor": actor,
                        "emotion": emotion,
                        "level": level
                    })


                except Exception as e:

                    print(f"Error processing {video_path}: {e}")


# ===============================
# Save labels CSV
# ===============================

df = pd.DataFrame(metadata)

df.to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)

print("✔ MFCC extraction complete!")
print(f"Total samples: {len(df)}")


# ===============================
# Cleanup
# ===============================

if os.path.exists(TEMP_AUDIO):
    os.remove(TEMP_AUDIO)