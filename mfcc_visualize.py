import matplotlib.pyplot as plt
import numpy as np
import os

mfcc_dir = "audio_mfcc"
mfcc_files = [f for f in os.listdir(mfcc_dir) if f.endswith(".npy")][:1]  # First 6 files

plt.figure(figsize=(15, 10))

for idx, filename in enumerate(mfcc_files):
    mfcc = np.load(os.path.join(mfcc_dir, filename))

    plt.subplot(1, 1, idx + 1)
    plt.imshow(mfcc.T, aspect='auto', origin='lower', cmap='magma')
    plt.title(filename)
    plt.xlabel("Frames")
    plt.ylabel("Coefficients")

plt.tight_layout()
plt.show()
