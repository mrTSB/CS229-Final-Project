#!/usr/bin/env python3
"""
DeepRSM/ConvLSTM Training Script for CTM Data

- Reads kernelized_tensor.npy with shape (5,2,288,600,1200).
- Downsamples to 75×150, applies log1p.
- Builds rolling-window sequences (context_window=24).
- Splits into 80/10/10 train/val/test.
- Trains a ConvLSTM model to predict the 5th channel.
- Uses a dynamic LR schedule (linear warmup 20%, then cosine half-wave).
- Plots train/val MSE + LR schedule.
- Evaluates on the test set, then plots **all** test samples in order, 
  with a color scale determined by each sample's predicted image (no white spots).
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# Define a custom red-based colormap (light to dark red)
red_gradient = LinearSegmentedColormap.from_list(
    "red_gradient",
    [
        "#fff5f5",  # Very light red/pink
        "#ffcfcf",  # Light pink
        "#ff9999",  # Soft red
        "#ff6666",  # Medium red
        "#ff3333",  # Bright red
        "#cc0000"   # Deep red (lighter than #4d0000)
    ]
)


# ---------------------------------------
# Hyperparameters
# ---------------------------------------
base_lr         = 3e-3
weight_decay    = 1e-4
num_epochs      = 100
batch_size      = 32
context_window  = 24
forecast_horizon= 1
warmup_ratio    = 0.2
target_h, target_w = 150, 300  # Downsampled resolution
output_channels = 5           # Channels in the downsampled data

# ---------------------------------------
# Device Configuration
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

# Current directory for saving outputs
current_dir = os.getcwd()

# ---------------------------------------
# Load Memory-Mapped Tensor
# ---------------------------------------
super_tensor_file = os.path.join(current_dir, "kernelized_tensor.npy")
kernelized_tensor = np.memmap(super_tensor_file, dtype="float32", mode="r", 
                              shape=(5, 2, 288, 600, 1200))
print("Loaded data shape:", kernelized_tensor.shape)

# ---------------------------------------
# Data Preprocessing (Downsample to target_h x target_w)
# ---------------------------------------
# Original spatial dimensions
orig_h, orig_w = 600, 1200
# Compute downsampling factors based on target resolution
factor_h = orig_h // target_h  # e.g., 600 // 75 = 8
factor_w = orig_w // target_w  # e.g., 1200 // 150 = 8

# Optionally, verify that the factors evenly divide the dimensions
assert target_h * factor_h == orig_h, "target_h does not evenly divide original height"
assert target_w * factor_w == orig_w, "target_w does not evenly divide original width"

num_time = 288
data_emissions = kernelized_tensor[:, 0, :, :, :]  # "emi"
emissions_data = np.transpose(data_emissions, (1, 0, 2, 3)).astype(np.float32)  # (288,5,600,1200)
Y_full = np.transpose(emissions_data, (0, 2, 3, 1))  # (288,600,1200,5)

Y_low = np.empty((num_time, target_h, target_w, output_channels), dtype=np.float32)
for i in range(num_time):
    # Reshape to (target_h, factor_h, target_w, factor_w, output_channels) then average over factor_h and factor_w
    Y_low[i] = Y_full[i].reshape(target_h, factor_h, target_w, factor_w, output_channels).mean(axis=(1,3))

# Apply log1p transformation
Y_low_log = np.log1p(Y_low)  # (288, target_h, target_w, 5)


# ---------------------------------------
# Build Rolling Windows for Sequences
# ---------------------------------------
def create_sequences(data, window_size):
    T = data.shape[0]
    X_list, y_list = [], []
    for i in range(T - window_size - forecast_horizon + 1):
        X_seq = data[i : i + window_size]
        y_seq = data[i + window_size : i + window_size + forecast_horizon]
        X_list.append(X_seq)
        y_list.append(y_seq[0])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# 80/10/10 Split
T_total = Y_low_log.shape[0]
train_size = int(0.8 * T_total)
val_size   = int(0.1 * T_total)
test_size  = T_total - train_size - val_size

Y_train_raw = Y_low_log[:train_size]
Y_val_raw   = Y_low_log[train_size:train_size+val_size]
Y_test_raw  = Y_low_log[train_size+val_size:]

X_train, Y_train_seq = create_sequences(Y_train_raw, context_window)
X_val,   Y_val_seq   = create_sequences(Y_val_raw,   context_window)
X_test,  Y_test_seq  = create_sequences(Y_test_raw,  context_window)

# Use all 5 channels as input; target is channel 4
X_train_raw = X_train
Y_train_raw = Y_train_seq[..., 4:]  # shape: (N_train,75,150,1)
X_val_raw   = X_val
Y_val_raw   = Y_val_seq[..., 4:]
X_test_raw  = X_test
Y_test_raw  = Y_test_seq[..., 4:]

print("Training sequences: X_train_raw:", X_train_raw.shape, 
      "Y_train_raw:", Y_train_raw.shape)
print("Validation sequences: X_val_raw:", X_val_raw.shape, 
      "Y_val_raw:", Y_val_raw.shape)
print("Test sequences: X_test_raw:", X_test_raw.shape, 
      "Y_test_raw:", Y_test_raw.shape)

# ---------------------------------------
# Normalize Data (using Training Set Stats)
# ---------------------------------------
X_train_mean = np.mean(X_train_raw, axis=(0,1,2,3), keepdims=True)
X_train_std  = np.std(X_train_raw,  axis=(0,1,2,3), keepdims=True)
Y_train_mean = np.mean(Y_train_raw, axis=(0,1,2,3), keepdims=True)
Y_train_std  = np.std(Y_train_raw,  axis=(0,1,2,3), keepdims=True)

X_train_norm = (X_train_raw - X_train_mean) / X_train_std
Y_train_norm = (Y_train_raw - Y_train_mean) / Y_train_std
X_val_norm   = (X_val_raw   - X_train_mean) / X_train_std
Y_val_norm   = (Y_val_raw   - Y_train_mean) / Y_train_std
X_test_norm  = (X_test_raw  - X_train_mean) / X_train_std
Y_test_norm  = (Y_test_raw  - Y_train_mean) / Y_train_std

# Convert to Tensors
X_train_norm = X_train_norm.transpose(0,1,4,2,3)  # -> (N_train,24,5,75,150)
Y_train_norm = Y_train_norm.transpose(0,3,1,2)    # -> (N_train,1,75,150)
train_dataset= TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32, device=device),
                             torch.tensor(Y_train_norm, dtype=torch.float32, device=device))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X_val_tensor = torch.tensor(X_val_norm.transpose(0,1,4,2,3), dtype=torch.float32, device=device)
Y_val_tensor = torch.tensor(Y_val_norm.transpose(0,3,1,2),   dtype=torch.float32, device=device)
X_test_tensor= torch.tensor(X_test_norm.transpose(0,1,4,2,3),dtype=torch.float32, device=device)
Y_test_tensor= torch.tensor(Y_test_norm.transpose(0,3,1,2),  dtype=torch.float32, device=device)

# ---------------------------------------
# Define the ConvLSTM Model
# ---------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4*hidden_channels,
                              kernel_size=kernel_size, padding=padding)
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f*c_cur + i*g
        h_next = o*torch.tanh(c_next)
        return h_next, c_next
    def init_hidden(self, batch_size, shape):
        height, width = shape
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, padding):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, padding)
    def forward(self, input_tensor):
        batch_size, seq_len, channels, H, W = input_tensor.shape
        h, c = self.cell.init_hidden(batch_size, (H, W))
        for t in range(seq_len):
            h, c = self.cell(input_tensor[:, t], (h, c))
        return h

class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, padding, dropout_p=0.2):
        super(ConvLSTMModel, self).__init__()
        self.convlstm = ConvLSTM(input_channels, hidden_channels, kernel_size, padding)
        self.conv1 = nn.Conv2d(hidden_channels, 16, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(p=dropout_p)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_p)
        self.conv3 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.convlstm(x)  # (batch, hidden_channels, H, W)
        out = self.relu(self.conv1(h))
        out = self.dropout1(out)
        out = self.relu(self.conv2(out))
        out = self.dropout2(out)
        out = self.conv3(out)
        return out

model = ConvLSTMModel(input_channels=5, hidden_channels=8, kernel_size=3, padding=1, dropout_p=0.2).to(device)
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(init_weights)
print(model)

optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

# ---------------------------------------
# Dynamic LR: Linear Warmup then Cosine Decay
# ---------------------------------------
# warmup_epochs = int(warmup_ratio * num_epochs)
# def lr_lambda(epoch):
#     if epoch < warmup_epochs:
#         return epoch / float(warmup_epochs)
#     else:
#         progress = (epoch - warmup_epochs) / float(num_epochs - warmup_epochs)
#         return 0.5 * (1 + math.cos(math.pi * progress))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ---------------------------------------
# Training Loop
# ---------------------------------------
train_mses = []
val_mses   = []
lr_history = []
for epoch in range(num_epochs):
    model.train()
    # One epoch of training
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
    # Step the scheduler after each epoch
    # scheduler.step()

    # Evaluate train loss over entire train set
    model.eval()
    epoch_loss_accum = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in train_loader:
            pred = model(batch_X)
            l = criterion(pred, batch_Y)
            epoch_loss_accum += l.item() * batch_X.size(0)
    epoch_loss_accum /= len(train_dataset)
    train_mses.append(epoch_loss_accum)

    # Validation MSE
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_loss  = criterion(val_preds, Y_val_tensor).item()
    val_mses.append(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, LR={current_lr:.2e}, "
              f"Train MSE={epoch_loss_accum:.6f}, Val MSE={val_loss:.6f}")

# ---------------------------------------
# Test Evaluation
# ---------------------------------------
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)  # shape: (N_test,1,75,150)

test_loss = criterion(test_outputs, Y_test_tensor).item()
print("Test MSE (normalized):", test_loss)

# Inverse normalization
Y_test_pred = test_outputs.cpu().numpy().reshape(-1, target_h, target_w, 1)
Y_test_true = Y_test_tensor.cpu().numpy().reshape(-1, target_h, target_w, 1)
Y_test_pred_log = Y_test_pred * Y_train_std + Y_train_mean
Y_test_true_log = Y_test_true * Y_train_std + Y_train_mean
# Inverse log1p
Y_test_pred_final = np.expm1(Y_test_pred_log)
Y_test_true_final = np.expm1(Y_test_true_log)
test_mse_final = mean_squared_error(Y_test_true_final.flatten(), Y_test_pred_final.flatten())
print("Test MSE (final):", test_mse_final)

# ---------------------------------------
# Plot LR, Train, and Validation Loss Curves
# ---------------------------------------
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
plt.savefig(os.path.join(current_dir, f"train_val_log_mse_and_lr_{base_lr}_{batch_size}_{num_epochs}.png"), dpi=300)
plt.show()


custom_multi_color = LinearSegmentedColormap.from_list(
    "custom_teal_yellow_8",
    [
        "#002b36",  # Dark teal
        "#0a484c",
        "#136664",
        "#1f847c",
        "#2ba294",
        "#49bfac",
        "#72dbbf",
        "#f0f921"   # Bright yellow
    ]
)

def plot_emission_comparison(tensor_gt, tensor_pred, idx, extent, cmap=custom_multi_color):
    """
    Plot ground truth and predicted emission for the 5th channel at a given index in the test set.
    The color scale is determined by the predicted image only.
    """
    gt_img   = tensor_gt[idx, :, :, 0]
    pred_img = tensor_pred[idx, :, :, 0]

    # 1) Clamp any zero or negative values to a small positive number for log scale
    gt_img_clamped   = np.where(gt_img <= 0, 1e-10, gt_img)
    pred_img_clamped = np.where(pred_img <= 0, 1e-10, pred_img)

    # 2) Determine the log range based on the predicted image
    pred_min = pred_img_clamped.min()
    pred_max = pred_img_clamped.max()
    # Avoid degenerate case if everything is the same
    if pred_min == pred_max:
        pred_max = pred_min + 1e-10

    # 3) Use a LogNorm over the predicted image range
    norm = LogNorm(vmin=pred_min, vmax=pred_max, clip=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # -- Ground Truth --
    im0 = axs[0].imshow(gt_img_clamped, cmap=cmap, interpolation='nearest',
                        extent=extent, norm=norm)
    fig.colorbar(im0, ax=axs[0], label="Log-Scaled Emissions (Pred-based)")
    axs[0].set_title(f"Ground Truth (Test idx {idx})")
    axs[0].set_xlabel("Longitude Grid Points")
    axs[0].set_ylabel("Latitude Grid Points")
    axs[0].invert_yaxis()

    # -- Prediction --
    im1 = axs[1].imshow(pred_img_clamped, cmap=cmap, interpolation='nearest',
                        extent=extent, norm=norm)
    fig.colorbar(im1, ax=axs[1], label="Log-Scaled Emissions (Pred-based)")
    axs[1].set_title(f"Prediction (Test idx {idx})")
    axs[1].set_xlabel("Longitude Grid Points")
    axs[1].set_ylabel("Latitude Grid Points")
    axs[1].invert_yaxis()

    plt.tight_layout()
    return fig


extent = [0, target_w, 0, target_h]
N_test = Y_test_true_final.shape[0]
# Plot all test samples in order
for i in range(N_test):
    fig = plot_emission_comparison(Y_test_true_final, Y_test_pred_final, i, extent)
    fig.savefig(os.path.join(current_dir, f"test_sample_{i}_{base_lr}_{batch_size}_{num_epochs}_{warmup_ratio}.png"), dpi=300)
    plt.close(fig)

print("Training, validation, and test evaluation complete.")
print("Final Train MSE:", train_mses[-1])
print("Final Val MSE:", val_mses[-1])
print("Final Test MSE (final):", test_mse_final)
print(f"{N_test} PNG files (one for each test sample) have been saved.")
