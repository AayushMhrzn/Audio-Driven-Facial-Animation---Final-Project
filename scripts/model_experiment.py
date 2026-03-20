import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from Model import CNN_TCN_Seq
import matplotlib.pyplot as plt
import pandas as pd

# ===========================
# General Parameters
# ===========================
WINDOW_SIZE = 20
OUT_LEN = 5
BATCH = 64
VAL_RATIO = 0.2
LR = 1e-3
LAMBDA_TEMP = 0.1
OUTPUT_DIR = "training_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# Load Data
# ===========================
X = np.load(os.path.join(OUTPUT_DIR, "X_seq.npy"))
Y = np.load(os.path.join(OUTPUT_DIR, "Y_centered_seq.npy"))

X_t = torch.tensor(X, dtype=torch.float32)
Y_t = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_t, Y_t)
val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, drop_last=False)

# ===========================
# Loss Functions
# ===========================
mse = nn.MSELoss()
def velocity_loss(pred, target):
    pdiff = pred[:, 1:, :] - pred[:, :-1, :]
    tdiff = target[:, 1:, :] - target[:, :-1, :]
    return mse(pdiff, tdiff)

# ===========================
# Hyperparameter Experiments Table
# ===========================
results = []

# ---------------------------
# Phase 1: Epoch Tuning
# ---------------------------
epoch_options = [10, 25, 40]
dropout_default = 0.2
tcn_blocks_default = [128, 128, 128, 128]

for epochs in epoch_options:
    print(f"\n🔹 Phase 1: Epoch Tuning → Epochs={epochs}")
    # Initialize model
    model = CNN_TCN_Seq(mfcc_dim=13, lip_points=20, out_len=OUT_LEN, tcn_channels=tcn_blocks_default)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loss_hist, val_loss_hist = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb) + LAMBDA_TEMP * velocity_loss(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train = total_train_loss / len(train_loader)
        train_loss_hist.append(avg_train)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = mse(pred, yb) + LAMBDA_TEMP * velocity_loss(pred, yb)
                total_val_loss += loss.item()
        avg_val = total_val_loss / len(val_loader)
        val_loss_hist.append(avg_val)

        print(f"Epoch {epoch+1}/{epochs}  Train: {avg_train:.6f}  Val: {avg_val:.6f}")

    results.append({
        "Phase": "Epoch",
        "Epochs": epochs,
        "Dropout": dropout_default,
        "Blocks": len(tcn_blocks_default),
        "Final_Train_Loss": train_loss_hist[-1],
        "Final_Val_Loss": val_loss_hist[-1],
        "Train_Loss_Hist": train_loss_hist,
        "Val_Loss_Hist": val_loss_hist
    })

# ---------------------------
# Phase 2: Dropout Tuning
# ---------------------------
best_epoch_exp = min([r for r in results if r["Phase"]=="Epoch"], key=lambda x: x["Final_Val_Loss"])
best_epochs = best_epoch_exp["Epochs"]
dropout_options = [0.1, 0.2, 0.3]

for dropout in dropout_options:
    print(f"\n🔹 Phase 2: Dropout Tuning → Dropout={dropout}")
    model = CNN_TCN_Seq(mfcc_dim=13, lip_points=20, out_len=OUT_LEN,
                        tcn_channels=tcn_blocks_default)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loss_hist, val_loss_hist = [], []

    for epoch in range(best_epochs):
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb) + LAMBDA_TEMP * velocity_loss(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train = total_train_loss / len(train_loader)
        train_loss_hist.append(avg_train)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = mse(pred, yb) + LAMBDA_TEMP * velocity_loss(pred, yb)
                total_val_loss += loss.item()
        avg_val = total_val_loss / len(val_loader)
        val_loss_hist.append(avg_val)

        print(f"Epoch {epoch+1}/{best_epochs}  Train: {avg_train:.6f}  Val: {avg_val:.6f}")

    results.append({
        "Phase": "Dropout",
        "Epochs": best_epochs,
        "Dropout": dropout,
        "Blocks": len(tcn_blocks_default),
        "Final_Train_Loss": train_loss_hist[-1],
        "Final_Val_Loss": val_loss_hist[-1],
        "Train_Loss_Hist": train_loss_hist,
        "Val_Loss_Hist": val_loss_hist
    })

# ---------------------------
# Phase 3: TCN Blocks Tuning
# ---------------------------
best_dropout_exp = min([r for r in results if r["Phase"]=="Dropout"], key=lambda x: x["Final_Val_Loss"])
best_dropout = best_dropout_exp["Dropout"]
blocks_options = [[128,128,128], [128,128,128,128]]

for blocks in blocks_options:
    print(f"\n🔹 Phase 3: TCN Blocks Tuning → Blocks={len(blocks)}")
    model = CNN_TCN_Seq(mfcc_dim=13, lip_points=20, out_len=OUT_LEN,
                        tcn_channels=blocks)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loss_hist, val_loss_hist = [], []

    for epoch in range(best_epochs):
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb) + LAMBDA_TEMP * velocity_loss(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train = total_train_loss / len(train_loader)
        train_loss_hist.append(avg_train)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = mse(pred, yb) + LAMBDA_TEMP * velocity_loss(pred, yb)
                total_val_loss += loss.item()
        avg_val = total_val_loss / len(val_loader)
        val_loss_hist.append(avg_val)

        print(f"Epoch {epoch+1}/{best_epochs}  Train: {avg_train:.6f}  Val: {avg_val:.6f}")

    results.append({
        "Phase": "Blocks",
        "Epochs": best_epochs,
        "Dropout": best_dropout,
        "Blocks": len(blocks),
        "Final_Train_Loss": train_loss_hist[-1],
        "Final_Val_Loss": val_loss_hist[-1],
        "Train_Loss_Hist": train_loss_hist,
        "Val_Loss_Hist": val_loss_hist
    })

# ===========================
# Save All Results Table
# ===========================
df = pd.DataFrame(results)
df.to_csv("hyperparam_full_experiments.csv", index=False)
print("✅ All hyperparameter experiment results saved!")

# ===========================
# Plot Best Experiment Curve
# ===========================
best_exp = df.loc[df["Final_Val_Loss"].idxmin()]
plt.figure(figsize=(8,5))
plt.plot(range(1, best_exp["Epochs"]+1), best_exp["Train_Loss_Hist"], marker='o', label='Train Loss')
plt.plot(range(1, best_exp["Epochs"]+1), best_exp["Val_Loss_Hist"], marker='s', label='Validation Loss')
plt.title(f"Best Train/Validation Loss Curve ({best_exp['Phase']})")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE + Velocity)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"best_loss_curve_{best_exp['Phase']}.png", dpi=300)
plt.show()
