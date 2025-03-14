#!/usr/bin/env python3
"""
EmissionNetV4 - Improved Transformer-Based Emission Prediction with Advanced Training Pipeline

- Model: Integrates an initial convolution, residual blocks, and a multi-head Transformer decoder.
- Uses mixed precision training with automatic loss scaling.
- Training uses DataLoader, dynamic LR scheduling (linear warmup then cosine decay), and loss tracking.
- Evaluation and plotting of training curves, predictions, and an autoregressive test evaluation are included.
"""

import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# Optionally, set an environment variable to help with fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------------------------------------------------------------------------
# 1) Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

# ----------------------------------------------------------------------------
# 2) Hyperparameters
# ----------------------------------------------------------------------------
base_lr         = 3e-3  
weight_decay    = 1e-4
num_epochs      = 5000
batch_size      = 8
context_window  = 24
forecast_horizon= 1      # used in sliding-window creation
target_h, target_w = 150, 300
output_channels = 5  
warmup_ratio    = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()
current_dir = os.getcwd()

# ----------------------------------------------------------------------------
# 3) Load Dataset and Downsample
# ----------------------------------------------------------------------------
super_tensor_file = os.path.join(current_dir, "kernelized_tensor.npy")
if not os.path.exists(super_tensor_file):
    raise FileNotFoundError(f"kernelized_tensor.npy not found in {current_dir}.")

# Load memory-mapped tensor (shape: (5,2,288,600,1200))
kernelized_tensor = np.memmap(super_tensor_file, dtype="float32", mode="r", 
                              shape=(5, 2, 288, 600, 1200))
num_time = 288
data_emissions = kernelized_tensor[:, 0, :, :, :]
emissions_data = np.transpose(data_emissions, (1, 0, 2, 3)).astype(np.float32)
Y_full = np.transpose(emissions_data, (0, 2, 3, 1))  # shape: (288,600,1200,5)

# Downsample spatially from (600,1200) to (150,300)
Y_low = np.empty((num_time, target_h, target_w, output_channels), dtype=np.float32)
factor_h, factor_w = 600 // target_h, 1200 // target_w
for i in range(num_time):
    Y_low[i] = Y_full[i].reshape(target_h, factor_h, target_w, factor_w, output_channels).mean(axis=(1,3))
Y_low_log = np.log1p(Y_low)

# ----------------------------------------------------------------------------
# 4) Create Sliding Windows (with forecast_horizon = 1)
# ----------------------------------------------------------------------------
def create_sequences(data, window_size, forecast_horizon):
    T = data.shape[0]
    X_list, y_list = [], []
    for i in range(T - window_size - forecast_horizon + 1):
        X_seq = data[i : i + window_size]
        y_seq = data[i + window_size : i + window_size + forecast_horizon]
        X_list.append(X_seq)
        y_list.append(y_seq[0])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

T_total = Y_low_log.shape[0]
train_size = int(0.8 * T_total)
val_size   = int(0.1 * T_total)
test_size  = T_total - train_size - val_size

# Create train, validation, and test segments ensuring overlapping windows for continuity.
train_segment = Y_low_log[:train_size]
X_train_raw, Y_train_seq = create_sequences(train_segment, context_window, forecast_horizon)
Y_train_raw = Y_train_seq[..., 4:]  # using channel index 4

val_segment = Y_low_log[train_size - context_window : train_size + val_size]
X_val_raw, Y_val_seq = create_sequences(val_segment, context_window, forecast_horizon)
Y_val_raw = Y_val_seq[..., 4:]

# For autoregressive evaluation later we also save the entire test segment:
test_segment = Y_low_log[train_size + val_size - context_window : T_total]
X_test_raw, Y_test_seq = create_sequences(test_segment, context_window, forecast_horizon)
Y_test_raw = Y_test_seq[..., 4:]

print("Training windows: X_train:", X_train_raw.shape, "Y_train:", Y_train_raw.shape)
print("Validation windows: X_val:", X_val_raw.shape, "Y_val:", Y_val_raw.shape)
print("Test windows: X_test:", X_test_raw.shape, "Y_test:", Y_test_raw.shape)

# ----------------------------------------------------------------------------
# 5) Normalize Data (using training set statistics)
# ----------------------------------------------------------------------------
X_train_mean = np.mean(X_train_raw, axis=(0,1,2,3), keepdims=True)
X_train_std  = np.std(X_train_raw, axis=(0,1,2,3), keepdims=True)
Y_train_mean = np.mean(Y_train_raw, axis=(0,1,2,3), keepdims=True)
Y_train_std  = np.std(Y_train_raw, axis=(0,1,2,3), keepdims=True)

X_train_norm = (X_train_raw - X_train_mean) / (X_train_std + 1e-8)
Y_train_norm = (Y_train_raw - Y_train_mean) / (Y_train_std + 1e-8)
X_val_norm   = (X_val_raw - X_train_mean) / (X_train_std + 1e-8)
Y_val_norm   = (Y_val_raw - Y_train_mean) / (Y_train_std + 1e-8)
X_test_norm  = (X_test_raw - X_train_mean) / (X_train_std + 1e-8)
Y_test_norm  = (Y_test_raw - Y_train_mean) / (Y_train_std + 1e-8)

# ----------------------------------------------------------------------------
# 6) Reshape Data for MLP Processing
# ----------------------------------------------------------------------------
# Our sliding window outputs:
#   X: (N, context_window, H, W, channels)
#   Y: (N, H, W, 1)
# For the MLP, we need:
#   X: (N, context_window, channels, H, W)
#   Y: (N, 1, H, W)
X_train_norm = X_train_norm.transpose(0, 1, 4, 2, 3)  # (N,24,5,150,300)
X_val_norm   = X_val_norm.transpose(0, 1, 4, 2, 3)
X_test_norm  = X_test_norm.transpose(0, 1, 4, 2, 3)
Y_train_norm = Y_train_norm.transpose(0, 3, 1, 2)      # (N,1,150,300)
Y_val_norm   = Y_val_norm.transpose(0, 3, 1, 2)
Y_test_norm  = Y_test_norm.transpose(0, 3, 1, 2)

print("After transpose, X_train_norm shape:", X_train_norm.shape)
print("After transpose, Y_train_norm shape:", Y_train_norm.shape)

# Convert to tensors.
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
Y_train_tensor = torch.tensor(Y_train_norm, dtype=torch.float32, device=device)
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)

X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
Y_val_tensor = torch.tensor(Y_val_norm, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
Y_test_tensor = torch.tensor(Y_test_norm, dtype=torch.float32, device=device)

# ----------------------------------------------------------------------------
# 7) Define Improved MLP Baseline Model (Per-Pixel) - MLP Baseline remains unchanged
# ----------------------------------------------------------------------------
class MLPModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_hidden=2, out_dim=1):
        super(MLPModel, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        # x shape: (batch, context_window, channels, H, W)
        N, T, C, H, W = x.size()
        x = x.permute(0, 3, 4, 1, 2)  # (N, H, W, T, C)
        x = x.reshape(N, H, W, T * C)  # (N, H, W, T*C)
        x = x.reshape(-1, T * C)       # (N*H*W, T*C)
        out = self.model(x)
        out = out.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        return out

# For our data, per-pixel input dimension = context_window * channels = 24 * 5 = 120.
input_dim = context_window * 5
model = MLPModel(in_dim=input_dim, hidden_dim=256, num_hidden=2, out_dim=1).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(init_weights)
print(model)

optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

# ----------------------------------------------------------------------------
# Dynamic LR: Linear Warmup then Cosine Decay
# ----------------------------------------------------------------------------
warmup_epochs = int(warmup_ratio * num_epochs)
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / float(warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / float(num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ----------------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------------
train_mses = []
val_mses   = []
lr_history = []

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    model.eval()
    epoch_loss_accum = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in train_loader:
            pred = model(batch_X)
            l = criterion(pred, batch_Y)
            epoch_loss_accum += l.item() * batch_X.size(0)
    epoch_loss_accum /= len(train_dataset)
    train_mses.append(epoch_loss_accum)
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_loss  = criterion(val_preds, Y_val_tensor).item()
    val_mses.append(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, LR={current_lr:.2e}, Train MSE={epoch_loss_accum:.6f}, Val MSE={val_loss:.6f}")

# ----------------------------------------------------------------------------
# Test Evaluation
# ----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
test_loss = criterion(test_preds, Y_test_tensor).item()
print("Test MSE (normalized):", test_loss)

# Inverse normalization for visualization
Y_test_pred = test_preds.cpu().numpy().reshape(-1, target_h, target_w, 1)
Y_test_true = Y_test_tensor.cpu().numpy().reshape(-1, target_h, target_w, 1)
Y_test_pred_log = Y_test_pred * Y_train_std + Y_train_mean
Y_test_true_log = Y_test_true * Y_train_std + Y_train_mean
Y_test_pred_final = np.expm1(Y_test_pred_log)
Y_test_true_final = np.expm1(Y_test_true_log)
test_mse_final = mean_squared_error(Y_test_true_final.flatten(), Y_test_pred_final.flatten())
print("Test MSE (final):", test_mse_final)

# ----------------------------------------------------------------------------
# Plot LR, Train, and Validation Loss Curves
# ----------------------------------------------------------------------------
epochs_arr = np.arange(num_epochs)
lr_plot = np.array(lr_history)
plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.plot(epochs_arr, np.log(train_mses), label='Log Train MSE')
plt.plot(epochs_arr, np.log(val_mses), label='Log Val MSE')
plt.xlabel('Epoch')
plt.ylabel('Log MSE')
plt.title('Train/Val MSE')
plt.legend()
plt.subplot(2,1,2)
plt.plot(epochs_arr, lr_plot, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.title('Dynamic Learning Rate')
plt.legend()
plt.tight_layout()
loss_plot_path = os.path.join(current_dir, f"{__file__}_train_val_log_mse_and_lr_{base_lr}_{batch_size}_{num_epochs}.png")
plt.savefig(loss_plot_path, dpi=300)
plt.show()

# ----------------------------------------------------------------------------
# Autoregressive Evaluation on the Test Set
# ----------------------------------------------------------------------------
# For autoregressive evaluation, we use the entire test segment (from step 4)
# We assume test_segment (of shape [T_test, H, W, 5]) is our continuous test series.
# We will predict the target channel (index 4) iteratively.
# For each new timestep, we replace the target channel in the window with the prediction
# while keeping the other channels (0-3) from ground truth.

X_train_mean_tensor = torch.tensor(X_train_mean, device=device)
X_train_std_tensor  = torch.tensor(X_train_std, device=device)

# Convert test_segment to a torch tensor.
# Note: test_segment was defined earlier and has shape (T_test, target_h, target_w, 5)
test_seq = torch.tensor(test_segment, dtype=torch.float32, device=device)  # shape: (T_test, 150, 300, 5)

# Normalize test_seq using training statistics.
# For simplicity, assume the same X_train_mean and X_train_std apply to all channels.
test_seq_norm = (test_seq - X_train_mean_tensor.squeeze(0)) / (X_train_std_tensor.squeeze(0) + 1e-8)

# Initialize the input window with the first context_window timesteps.
# Rearrange to match model input: (1, context_window, channels, H, W)
window = test_seq_norm[:context_window].permute(0, 3, 1, 2).unsqueeze(0)  # shape: (1, 24, 5, 150, 300)

autoregress_losses = []
autoregress_steps = []

# Iterate from timestep context_window to end of test sequence.
T_test = test_seq_norm.shape[0]
model.eval()
with torch.no_grad():
    for t in range(context_window, T_test):
        # Predict next timestep using current window.
        pred = model(window)  # shape: (1, 1, 150, 300)
        # Ground truth for target channel at timestep t: extract channel index 4.
        gt = test_seq_norm[t, :, :, 4].unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 150, 300)
        loss_step = criterion(pred, gt)
        autoregress_losses.append(loss_step.item())
        autoregress_steps.append(t)
        # Prepare new frame: use ground truth for channels 0-3 and predicted value for channel 4.
        new_frame = test_seq_norm[t].clone()  # shape: (150, 300, 5)
        new_frame[..., 4] = pred.squeeze(0).squeeze(0)  # replace target channel with prediction
        # Rearrange new_frame to (1, 5, 150, 300)
        new_frame = new_frame.permute(2, 0, 1).unsqueeze(0)
        # Update window: remove first timestep and append new_frame.
        window = torch.cat([window[:, 1:], new_frame.unsqueeze(1)], dim=1)

# Plot the autoregressive MSE loss over timesteps.
plt.figure(figsize=(8,6))
plt.plot(autoregress_steps, autoregress_losses, label="Autoregressive MSE Loss")
plt.xlabel("Timestep")
plt.ylabel("MSE Loss")
plt.title("Autoregressive Evaluation Loss over Test Timesteps")
plt.legend()
autoregress_plot_path = os.path.join(current_dir, f"{__file__}_autoregressive_loss_{base_lr}_{batch_size}_{num_epochs}.png")
plt.savefig(autoregress_plot_path, dpi=300)
plt.show()

print(autoregress_losses)

# ----------------------------------------------------------------------------
# Plot Predictions (Optional)
# ----------------------------------------------------------------------------
custom_multi_color = LinearSegmentedColormap.from_list(
    "custom_teal_yellow_8",
    [
        "#002b36",
        "#0a484c",
        "#136664",
        "#1f847c",
        "#2ba294",
        "#49bfac",
        "#72dbbf",
        "#f0f921"
    ]
)

def plot_emission_comparison(tensor_gt, tensor_pred, idx, extent, cmap=custom_multi_color):
    gt_img   = tensor_gt[idx, :, :, 0]
    pred_img = tensor_pred[idx, :, :, 0]
    gt_img_clamped   = np.where(gt_img <= 0, 1e-10, gt_img)
    pred_img_clamped = np.where(pred_img <= 0, 1e-10, pred_img)
    pred_min = gt_img_clamped.min()
    pred_max = gt_img_clamped.max()
    if pred_min == pred_max:
        pred_max = pred_min + 1e-10
    norm = LogNorm(vmin=pred_min, vmax=pred_max, clip=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs[0].imshow(gt_img_clamped, cmap=cmap, interpolation='nearest', extent=extent, norm=norm)
    fig.colorbar(im0, ax=axs[0], label="Log-Scaled Emissions")
    axs[0].set_title(f"Ground Truth (Test idx {idx})")
    axs[0].set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")
    axs[0].invert_yaxis()
    im1 = axs[1].imshow(pred_img_clamped, cmap=cmap, interpolation='nearest', extent=extent, norm=norm)
    fig.colorbar(im1, ax=axs[1], label="Log-Scaled Emissions")
    axs[1].set_title(f"Prediction (Test idx {idx})")
    axs[1].set_xlabel("Longitude")
    axs[1].set_ylabel("Latitude")
    axs[1].invert_yaxis()
    plt.tight_layout()
    return fig

extent = [0, target_w, 0, target_h]
N_test = Y_test_true_final.shape[0]
for i in range(N_test):
    fig = plot_emission_comparison(Y_test_true_final, Y_test_pred_final, i, extent)
    fig_path = os.path.join(current_dir, f"{__file__}_test_sample_{i}_{base_lr}_{batch_size}_{num_epochs}.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------
# Save Model and Final Logs
# ----------------------------------------------------------------------------
torch.save(model.state_dict(), "emissionnet_v4.pth")
print("Training, validation, and test evaluation complete.")
print("Final Train MSE:", train_mses[-1])
print("Final Val MSE:", val_mses[-1])
print("Final Test MSE (final):", test_mse_final)
print(f"{N_test} PNG files (one for each test sample) have been saved.")
