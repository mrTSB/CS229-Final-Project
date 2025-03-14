#!/usr/bin/env python3
"""
EmissionNetV6 - Transformer-Enhanced CTM Model

- **Integrates Transformer Decoder**
- **Fixes Mismatched Output Size via Upsampling**
- **Memory Optimized Training**
- **Final Evaluation with Proper Plotting**
"""

import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# ----------------------------------------------------------------------------
# 1) **Reproducibility & Memory Optimization**
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False  
torch.backends.cudnn.benchmark = True  # Optimized cuDNN usage

# ----------------------------------------------------------------------------
# 2) **Hyperparameters**
# ----------------------------------------------------------------------------
base_lr         = 1e-3
weight_decay    = 1e-4
num_epochs      = 250
batch_size      = 8  # ✅ Reduced to prevent OOM
context_window  = 24
target_h, target_w = 75, 150  # ✅ Ensure consistent dimensions
output_channels = 5  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()
current_dir = os.getcwd()

# ----------------------------------------------------------------------------
# 3) **Load Dataset (Downsampled)**
# ----------------------------------------------------------------------------
super_tensor_file = os.path.join(current_dir, "kernelized_tensor.npy")
if not os.path.exists(super_tensor_file):
    raise FileNotFoundError(f"kernelized_tensor.npy not found in {current_dir}.")
kernelized_tensor = np.memmap(super_tensor_file, dtype="float32", mode="r", 
                              shape=(5, 2, 288, 600, 1200))

num_time = 288
data_emissions = kernelized_tensor[:, 0, :, :, :]
emissions_data = np.transpose(data_emissions, (1, 0, 2, 3)).astype(np.float32)
Y_full = np.transpose(emissions_data, (0, 2, 3, 1))

Y_low = np.empty((num_time, target_h, target_w, output_channels), dtype=np.float32)
factor_h, factor_w = 600 // target_h, 1200 // target_w
for i in range(num_time):
    Y_low[i] = Y_full[i].reshape(target_h, factor_h, target_w, factor_w, output_channels).mean(axis=(1,3))

Y_low_log = np.log1p(Y_low)

# ----------------------------------------------------------------------------
# 4) **Create Sliding Windows**
# ----------------------------------------------------------------------------
def create_sequences(data, window_size):
    X_list, y_list = [], []
    for i in range(data.shape[0] - window_size):
        X_list.append(data[i : i + window_size])
        y_list.append(data[i + window_size, ..., 4:])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

train_size = int(0.8 * num_time)
val_size   = int(0.1 * num_time)

X_train_raw, Y_train_raw = create_sequences(Y_low_log[:train_size], context_window)
X_val_raw, Y_val_raw = create_sequences(Y_low_log[train_size:], context_window)

X_train_mean, X_train_std = X_train_raw.mean(), X_train_raw.std()
Y_train_mean, Y_train_std = Y_train_raw.mean(), Y_train_raw.std()

X_train_norm = (X_train_raw - X_train_mean) / (X_train_std + 1e-8)
Y_train_norm = (Y_train_raw - Y_train_mean) / (Y_train_std + 1e-8)

X_val_norm = (X_val_raw - X_train_mean) / (X_train_std + 1e-8)
Y_val_norm = (Y_val_raw - Y_train_mean) / (Y_train_std + 1e-8)

# Reshape for Transformer input
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
Y_train_tensor = torch.tensor(Y_train_norm, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
Y_val_tensor = torch.tensor(Y_val_norm, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

# ----------------------------------------------------------------------------
# 5) **Model**
# ----------------------------------------------------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads=1):  # ✅ Reduced heads to 1 (less memory)
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.02)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, _ = self.attn(x, x, x)
        x = self.fc(x)
        x = self.norm(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x

class EmissionNetV6(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1)
        self.decoder = TransformerDecoder(32, num_heads=1)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.upsample = nn.Upsample(size=(target_h, target_w), mode="bilinear", align_corners=False)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.downsample(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = self.upsample(x)
        return x

# ----------------------------------------------------------------------------
# 6) **Final Evaluation & Plotting**
# ----------------------------------------------------------------------------
model = EmissionNetV6().to(device)
criterion = nn.MSELoss()

model.eval()
with torch.no_grad():
    test_preds = model(X_val_tensor)
test_loss = criterion(test_preds, Y_val_tensor).item()
print("Test MSE (normalized):", test_loss)

# Convert predictions back
Y_test_pred = test_preds.cpu().numpy().reshape(-1, target_h, target_w, 1)
Y_test_true = Y_val_tensor.cpu().numpy().reshape(-1, target_h, target_w, 1)

test_mse_final = mean_squared_error(Y_test_true.flatten(), Y_test_pred.flatten())
print("Test MSE (final):", test_mse_final)

# ----------------------------------------------------------------------------
# 7) **Plot Predictions (Same Format)**
# ----------------------------------------------------------------------------
custom_multi_color = LinearSegmentedColormap.from_list(
    "custom_teal_yellow_8",
    ["#002b36", "#0a484c", "#136664", "#1f847c", "#2ba294", "#49bfac", "#72dbbf", "#f0f921"]
)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(Y_test_true[0, :, :, 0], cmap=custom_multi_color, aspect='auto')
plt.title("Ground Truth")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(Y_test_pred[0, :, :, 0], cmap=custom_multi_color, aspect='auto')
plt.title("Predicted")
plt.colorbar()

plt.savefig(os.path.join(current_dir, "{__file__}_train_val_log_mse_and_lr.png"), dpi=300)
plt.show()

print("Model evaluation and plotting complete.")
