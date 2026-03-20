import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from Model import CNN_TCN_Seq

# ===========================
# PARAMETERS
# ===========================
WINDOW_SIZE = 20
OUT_LEN = 5
BATCH = 64
VAL_RATIO = 0.2
LR = 1e-3
LAMBDA_TEMP = 0.1
EPOCHS = 30

OUTPUT_DIR = "training_data"
SAVE_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# LOAD DATA
# ===========================
X = np.load(os.path.join(OUTPUT_DIR, "combined_X_seq.npy"))
Y = np.load(os.path.join(OUTPUT_DIR, "combined_Y_seq.npy"))

X_t = torch.tensor(X, dtype=torch.float32)
Y_t = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_t, Y_t)

val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

# ===========================
# MODEL
# ===========================
model = CNN_TCN_Seq(
    mfcc_dim=13,
    lip_points=20,
    out_len=OUT_LEN,
    tcn_channels=[128,128,128,128]
)

model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
mse = nn.MSELoss()

# velocity loss
def velocity_loss(pred, target):
    pdiff = pred[:,1:,:] - pred[:,:-1,:]
    tdiff = target[:,1:,:] - target[:,:-1,:]
    return mse(pdiff, tdiff)

# ===========================
# TRAINING LOOP
# ===========================

best_val_loss = float("inf")

train_losses = []
val_losses = []

print("Training started...\n")

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    train_loss = 0

    for xb, yb in train_loader:

        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()

        pred = model(xb)

        loss = mse(pred, yb) + LAMBDA_TEMP * velocity_loss(pred, yb)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for xb, yb in val_loader:

            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            pred = model(xb)

            loss = mse(pred, yb) + LAMBDA_TEMP * velocity_loss(pred, yb)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train {train_loss:.6f} | Val {val_loss:.6f}")

    # ---- SAVE BEST MODEL ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print("✅ Best model saved!")

# save loss history
np.save("train_losses_combined.npy", np.array(train_losses))
np.save("val_losses_combined.npy", np.array(val_losses))

print("\nTraining complete.")
print("Best validation loss:", best_val_loss)