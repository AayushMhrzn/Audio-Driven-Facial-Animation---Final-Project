import numpy as np
import json
from tqdm import tqdm
from extract_lip_features import extract_features

#DATA_DIR = "training_data_seq_centered"
Y_PATH = "training_data/Y_seq.npy"

print("Loading training lip data...")
Y = np.load(Y_PATH)  
# shape: (N, OUT_LEN, 40)

N, OUT_LEN, _ = Y.shape
print(f"Loaded Y_centered_seq: {Y.shape}")

# storage
feature_values = {
    "mouth_open": [],
    "mouth_width": [],
    "upper_lip_raise": [],
    "lip_rounding": []
}

print("Extracting features from training data...")
for i in tqdm(range(N)):
    for t in range(OUT_LEN):
        lips_flat = Y[i, t]              # (40,)
        lips = lips_flat.reshape(20, 2)  # (20,2)

        feats = extract_features(lips)
        for k, v in feats.items():
            feature_values[k].append(v)

# compute stats
stats = {}
for k, vals in feature_values.items():
    vals = np.array(vals)
    stats[k] = {
        "min": float(vals.min()),
        "max": float(vals.max())
    }

# save
with open("feature_stats_b.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n✅ feature_stats.json created successfully:")
for k, v in stats.items():
    print(f"{k:16s} min={v['min']:.6f} max={v['max']:.6f}")




















