#!/usr/bin/env python3
"""
EmissionNetV4 - Transformer-Enhanced CTM Model with DataLoader Training, Mixed Precision, and Plotting

- Integrates Axial Attention and ResNet-style Bottleneck Blocks.
- Uses Residual Implicit Deep Supervision (R-IDS) with a Transformer Decoder.
- Trains with a DataLoader, dynamic LR scheduler (linear warmup then cosine decay), and plots training curves and predictions.
- Uses mixed precision training to reduce memory usage.
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
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

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
base_lr         = 1e-3
weight_decay    = 1e-4
num_epochs      = 200
batch_size      = 4  # Try reducing further if necessary (e.g., 4)
context_window  = 24
forecast_horizon= 1  # used in sliding-window creation
warmup_ratio    = 0.2
target_h, target_w = 150, 300
output_channels = 5  # channels in downsampled emissions

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

# Load memory-mapped tensor: shape (5,2,288,600,1200)
kernelized_tensor = np.memmap(super_tensor_file, dtype="float32", mode="r", 
                              shape=(5, 2, 288, 600, 1200))

num_time = 288
# Use the first channel along axis=1 (e.g. "emi")
data_emissions = kernelized_tensor[:, 0, :, :, :]
# Rearrange to (288, 5, 600, 1200)
emissions_data = np.transpose(data_emissions, (1, 0, 2, 3)).astype(np.float32)
# Rearrange to (288, 600, 1200, 5)
Y_full = np.transpose(emissions_data, (0, 2, 3, 1))

# Downsample spatially from (600,1200) to (150,300)
Y_low = np.empty((num_time, target_h, target_w, output_channels), dtype=np.float32)
factor_h, factor_w = 600 // target_h, 1200 // target_w
for i in range(num_time):
    Y_low[i] = Y_full[i].reshape(target_h, factor_h, target_w, factor_w, output_channels).mean(axis=(1,3))

Y_low_log = np.log1p(Y_low)

# ----------------------------------------------------------------------------
# 4) Create Sliding Windows using forecast_horizon = 1
# ----------------------------------------------------------------------------
def create_sequences(data, window_size):
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

# Training data
train_segment = Y_low_log[:train_size]
X_train_raw, Y_train_seq = create_sequences(train_segment, context_window)
# Extract the target channel (e.g., channel 4 for NO2)
Y_train_raw = Y_train_seq[..., 4:]

# Validation data: ensure overlapping windows for continuity
val_segment = Y_low_log[train_size - context_window : train_size + val_size]
X_val_raw, Y_val_seq = create_sequences(val_segment, context_window)
Y_val_raw = Y_val_seq[..., 4:]

# Test data
test_segment = Y_low_log[train_size + val_size - context_window : T_total]
X_test_raw, Y_test_seq = create_sequences(test_segment, context_window)
Y_test_raw = Y_test_seq[..., 4:]

print("Training windows: X_train:", X_train_raw.shape, "Y_train:", Y_train_raw.shape)
print("Validation windows: X_val:", X_val_raw.shape, "Y_val:", Y_val_raw.shape)
print("Test windows: X_test:", X_test_raw.shape, "Y_test:", Y_test_raw.shape)

# ----------------------------------------------------------------------------
# 5) Normalize Data (using training set stats)
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
# 6) Reshape Data for EmissionNetV4
# ----------------------------------------------------------------------------
# X: expected shape (N, context_window, H, W, channels)
# Permute to (N, context_window, channels, H, W)
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
X_val_tensor   = torch.tensor(X_val_norm, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
X_test_tensor  = torch.tensor(X_test_norm, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)

# Y: expected shape (N, H, W, 1) -> permute to (N, 1, H, W)
Y_train_tensor = torch.tensor(Y_train_norm, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
Y_val_tensor   = torch.tensor(Y_val_norm, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
Y_test_tensor  = torch.tensor(Y_test_norm, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

# ----------------------------------------------------------------------------
# 7) Define EmissionNetV4 Model
# ----------------------------------------------------------------------------
class AxialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0))

    def forward(self, x):
        # x shape: [batch, time, channels, height, width]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.conv1(x) + self.conv2(x)
        x = x.view(b, t, c, h, w)
        return x

class MFE_Module(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, 24, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, 24, kernel_size=5, padding=2)
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_channels, 24, kernel_size=1))
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
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

class EmissionNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.axial = AxialAttention(5)
        self.mfe = MFE_Module(5)
        self.downsample = nn.Conv2d(96, 64, kernel_size=3, stride=2, padding=1)
        self.decoder = TransformerDecoder(64, num_heads=4)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        # Add an upsampling layer to scale the output back to (150, 300)
        self.upsample = nn.Upsample(size=(150, 300), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.axial(x)
        x = x.mean(dim=1)  # Temporal aggregation
        x = self.mfe(x)
        x = self.downsample(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = self.upsample(x)  # Upsample to match target size
        return x


model = EmissionNetV4().to(device)
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
criterion = nn.MSELoss()
scaler = GradScaler()  # For mixed precision training

# ----------------------------------------------------------------------------
# 8) Setup LR Scheduler and DataLoader
# ----------------------------------------------------------------------------
warmup_epochs = int(warmup_ratio * num_epochs)
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / float(warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / float(num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)

# ----------------------------------------------------------------------------
# 9) Training Loop with Mixed Precision
# ----------------------------------------------------------------------------
train_mses, val_mses, lr_history = [], [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss_accum = 0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            preds = model(batch_X)
            loss = criterion(preds, batch_Y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss_accum += loss.item() * batch_X.size(0)
    epoch_loss_accum /= len(train_dataset)
    train_mses.append(epoch_loss_accum)
    
    with torch.no_grad():
        val_out = model(X_val_tensor)
        val_loss = criterion(val_out, Y_val_tensor).item()
    val_mses.append(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    scheduler.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, LR={current_lr:.2e}, Train MSE={epoch_loss_accum:.6f}, Val MSE={val_loss:.6f}")

# ----------------------------------------------------------------------------
# 10) Test Evaluation and Unnormalizing Predictions
# ----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
test_loss = criterion(test_preds, Y_test_tensor).item()
print("Test MSE (normalized):", test_loss)

Y_test_pred = test_preds.cpu().numpy().reshape(-1, target_h, target_w, 1)
Y_test_true = Y_test_tensor.cpu().numpy().reshape(-1, target_h, target_w, 1)
Y_test_pred_log = Y_test_pred * (Y_train_std + 1e-8) + Y_train_mean
Y_test_true_log = Y_test_true * (Y_train_std + 1e-8) + Y_train_mean
Y_test_pred_final = np.expm1(Y_test_pred_log)
Y_test_true_final = np.expm1(Y_test_true_log)
test_mse_final = mean_squared_error(Y_test_true_final.flatten(), Y_test_pred_final.flatten())
print("Test MSE (final):", test_mse_final)

# ----------------------------------------------------------------------------
# 11) Plot Loss Curves
# ----------------------------------------------------------------------------
epochs_arr = np.arange(num_epochs)
plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.plot(epochs_arr, np.log(train_mses), label='Log Train MSE')
plt.plot(epochs_arr, np.log(val_mses), label='Log Val MSE')
plt.xlabel('Epoch')
plt.ylabel('Log MSE')
plt.title('Train/Val MSE')
plt.legend()

plt.subplot(2,1,2)
plt.plot(epochs_arr, lr_history, label='Learning Rate')
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
plt.plot(list(range(T_test - context_window)), autoregress_losses, label="Autoregressive MSE Loss")
plt.xlabel("Timestep")
plt.ylabel("MSE Loss")
plt.title("Autoregressive Evaluation Loss over Test Timesteps")
plt.legend()
autoregress_plot_path = os.path.join(current_dir, f"{__file__}_autoregressive_loss_{base_lr}_{batch_size}_{num_epochs}.png")
plt.savefig(autoregress_plot_path, dpi=300)
plt.show()

print(autoregress_losses)
# ----------------------------------------------------------------------------
# 12) Plot Predictions (Optional)
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
N_test_samples = Y_test_true_final.shape[0]
for i in range(N_test_samples):
    fig = plot_emission_comparison(Y_test_true_final, Y_test_pred_final, i, extent)
    fig_path = os.path.join(current_dir, f"{__file__}_test_sample_{i}_{base_lr}_{batch_size}_{num_epochs}.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

print("Training, validation, and test evaluation complete.")
print("Final Train MSE:", train_mses[-1])
print("Final Val MSE:", val_mses[-1])
print("Final Test MSE (final):", test_mse_final)
print(f"{N_test_samples} PNG files (one for each test sample) have been saved.")
