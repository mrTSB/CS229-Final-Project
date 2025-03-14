#!/usr/bin/env python3
"""
EmissionNetV3 Training Script for CTM Data

- Loads kernelized_tensor.npy with shape (5,2,288,600,1200).
- Downsamples to 150×300, applies log1p.
- Creates rolling-window sequences (context_window=24).
- Splits data into train/val/test sets with overlap.
- Trains EmissionNetV3 to predict the 5th channel.
- Uses AdamW with a dynamic LR schedule (linear warmup then cosine decay).
- Evaluates and plots training curves.
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
num_epochs      = 200
batch_size      = 32
context_window  = 24
forecast_horizon= 1
warmup_ratio    = 0.2
target_h, target_w = 150, 300
output_channels = 5  # channels in original downsampled data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()
current_dir = os.getcwd()

# ----------------------------------------------------------------------------
# 3) Load Memory-Mapped Tensor
# ----------------------------------------------------------------------------
super_tensor_file = os.path.join(current_dir, "kernelized_tensor.npy")
if not os.path.exists(super_tensor_file):
    raise FileNotFoundError(f"kernelized_tensor.npy not found in {current_dir}.")
kernelized_tensor = np.memmap(super_tensor_file, dtype="float32", mode="r", 
                              shape=(5, 2, 288, 600, 1200))
print("Loaded data shape:", kernelized_tensor.shape)

# ----------------------------------------------------------------------------
# 4) Data Preprocessing (Downsample to 150×300, log1p)
# ----------------------------------------------------------------------------
orig_h, orig_w = 600, 1200
factor_h = orig_h // target_h  # e.g., 600//150 = 4
factor_w = orig_w // target_w  # e.g., 1200//300 = 4
assert target_h * factor_h == orig_h, "target_h does not evenly divide original height"
assert target_w * factor_w == orig_w, "target_w does not evenly divide original width"

num_time = 288
data_emissions = kernelized_tensor[:, 0, :, :, :]  # Use channel "emi"
# Rearrange to (288, 5, 600, 1200)
emissions_data = np.transpose(data_emissions, (1, 0, 2, 3)).astype(np.float32)
# Rearrange to (288, 600, 1200, 5)
Y_full = np.transpose(emissions_data, (0, 2, 3, 1))

Y_low = np.empty((num_time, target_h, target_w, output_channels), dtype=np.float32)
for i in range(num_time):
    Y_low[i] = Y_full[i].reshape(target_h, factor_h, target_w, factor_w, output_channels).mean(axis=(1,3))

Y_low_log = np.log1p(Y_low)  # shape: (288,150,300,5)

# ----------------------------------------------------------------------------
# 5) Create Sliding Windows (context_window=24)
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

train_segment = Y_low_log[:train_size]
X_train_raw, Y_train_seq = create_sequences(train_segment, context_window)
Y_train_raw = Y_train_seq[..., 4:]  # Use channel 4 for NO2

val_segment = Y_low_log[train_size - context_window : train_size + val_size]
X_val_raw, Y_val_seq = create_sequences(val_segment, context_window)
Y_val_raw = Y_val_seq[..., 4:]

test_segment = Y_low_log[train_size + val_size - context_window : T_total]
X_test_raw, Y_test_seq = create_sequences(test_segment, context_window)
Y_test_raw = Y_test_seq[..., 4:]

print("Training windows: X_train:", X_train_raw.shape, "Y_train:", Y_train_raw.shape)
print("Validation windows: X_val:", X_val_raw.shape, "Y_val:", Y_val_raw.shape)
print("Test windows: X_test:", X_test_raw.shape, "Y_test:", Y_test_raw.shape)

# ----------------------------------------------------------------------------
# 6) Normalize Data (using training set stats)
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
# 7) Reshape Data for EmissionNetV3
# ----------------------------------------------------------------------------
# Our sliding window outputs X: (N, context_window, H, W, channels)
# We flatten temporal and channel dims: (N, context_window*channels, H, W)
N_train = X_train_norm.shape[0]
X_train_norm = X_train_norm.reshape(N_train, context_window * 5, target_h, target_w)
N_val = X_val_norm.shape[0]
X_val_norm = X_val_norm.reshape(N_val, context_window * 5, target_h, target_w)
N_test = X_test_norm.shape[0]
X_test_norm = X_test_norm.reshape(N_test, context_window * 5, target_h, target_w)
# Targets Y: (N, H, W, 1) -> (N, 1, H, W)
Y_train_norm = Y_train_norm.transpose(0, 3, 1, 2)
Y_val_norm   = Y_val_norm.transpose(0, 3, 1, 2)
Y_test_norm  = Y_test_norm.transpose(0, 3, 1, 2)

print("After reshape, X_train_norm shape:", X_train_norm.shape)  # Expected: (N_train, 120, 150, 300)
print("After transpose, Y_train_norm shape:", Y_train_norm.shape)  # Expected: (N_train, 1, 150, 300)

# Convert to PyTorch tensors.
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
Y_train_tensor = torch.tensor(Y_train_norm, dtype=torch.float32, device=device)
X_val_tensor   = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
Y_val_tensor   = torch.tensor(Y_val_norm, dtype=torch.float32, device=device)
X_test_tensor  = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
Y_test_tensor  = torch.tensor(Y_test_norm, dtype=torch.float32, device=device)

# ----------------------------------------------------------------------------
# 8) Define Architecture Modules: MFE, ChannelAttention, BasicLayer, IDS
# ----------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BasicLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BasicLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.ca = ChannelAttention(growth_rate)
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.ca(out)
        return out

class ImplicitDeepSupervisionModule(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=6):
        super(ImplicitDeepSupervisionModule, self).__init__()
        self.basic_layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            self.basic_layers.append(BasicLayer(current_channels, growth_rate))
            current_channels += growth_rate
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(current_channels, in_channels, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        features = [x]
        for layer in self.basic_layers:
            merged = torch.cat(features, dim=1)
            new_feat = layer(merged)
            features.append(new_feat)
        combined = torch.cat(features, dim=1)
        out = self.dim_reduction(combined)
        return out

class MFE_Module(nn.Module):
    def __init__(self, in_channels, branch_channels):
        super(MFE_Module, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0)
        self.branch2 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0)
        )
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return out

# ----------------------------------------------------------------------------
# 9) Define Refined EmissionNetV3 Architecture
# ----------------------------------------------------------------------------
class EmissionNet(nn.Module):
    def __init__(self, input_channels=120, target_size=(150,300),
                 branch_channels=96, growth_rate=32):
        super(EmissionNet, self).__init__()
        self.target_size = target_size
        # Head Layers: increase capacity.
        self.head1 = nn.Conv2d(input_channels, 256, kernel_size=3, stride=2, padding=1)   # (N,256,75,150)
        self.head2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)               # (N,512,38,75)
        
        # 5 MFE Modules (sequential) operating on the head output.
        self.mfe1 = MFE_Module(512, branch_channels)    # out: 4*branch_channels = 384 channels
        self.mfe2 = MFE_Module(384, branch_channels)      # out: 384
        self.mfe3 = MFE_Module(384, branch_channels)      # out: 384
        self.mfe4 = MFE_Module(384, branch_channels)      # out: 384
        self.mfe5 = MFE_Module(384, branch_channels)      # out: 384
        
        # Implicit Deep Supervision Module: 6 layers, growth_rate=32
        self.idsm = ImplicitDeepSupervisionModule(in_channels=384,
                                                  growth_rate=growth_rate,
                                                  num_layers=6)
        # Skip connection from MFE output
        self.skip_conv = nn.Conv2d(384, 384, kernel_size=1)
        
        # Regression Head: Concatenate skip and IDS output
        # First, upsample IDS output to match skip's spatial dims.
        self.regress_conv1 = nn.Conv2d(384 * 2, 256, kernel_size=3, padding=1)
        self.regress_relu = nn.ReLU(inplace=True)
        self.regress_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.regress_conv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        # x: (N,120,150,300)
        x = self.head1(x)        # -> (N,256,75,150)
        x = self.head2(x)        # -> (N,512,38,75)
        
        x_mfe = self.mfe1(x)     # -> (N,384,38,75)
        x_mfe = self.mfe2(x_mfe) # -> (N,384,38,75)
        x_mfe = self.mfe3(x_mfe) # -> (N,384,38,75)
        x_mfe = self.mfe4(x_mfe) # -> (N,384,38,75)
        x_mfe = self.mfe5(x_mfe) # -> (N,384,38,75)
        
        # Save skip connection from MFE output.
        skip = self.skip_conv(x_mfe)  # (N,384,38,75)
        
        # IDS module processing.
        x_ids = self.idsm(x_mfe)      # -> (N,384, ~19, ~37)
        # Upsample IDS output to match skip spatial dims.
        x_ids_up = nn.functional.interpolate(x_ids, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate skip and IDS output along channel dimension.
        x_cat = torch.cat([skip, x_ids_up], dim=1)  # (N,768,38,75)
        
        # Regression head.
        x_reg = self.regress_conv1(x_cat)  # -> (N,256,38,75)
        x_reg = self.regress_relu(x_reg)
        x_reg = self.regress_conv2(x_reg)  # -> (N,128,38,75)
        x_reg = self.regress_conv3(x_reg)  # -> (N,1,38,75)
        x_out = self.upsample(x_reg)       # -> (N,1,150,300)
        return x_out

model = EmissionNet(input_channels=context_window * output_channels,
                    target_size=(target_h, target_w),
                    branch_channels=96,
                    growth_rate=32).to(device)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(init_weights)
print(model)

# ----------------------------------------------------------------------------
# 10) Optimizer, LR Scheduler, Dataloaders
# ----------------------------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

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

train_mses, val_mses, lr_history = [], [], []

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_Y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    model.eval()
    epoch_loss_accum = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in train_loader:
            out = model(batch_X)
            epoch_loss_accum += criterion(out, batch_Y).item() * batch_X.size(0)
    epoch_loss_accum /= len(train_dataset)
    train_mses.append(epoch_loss_accum)
    
    with torch.no_grad():
        val_out = model(X_val_tensor)
        val_loss = criterion(val_out, Y_val_tensor).item()
    val_mses.append(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, LR={current_lr:.2e}, "
              f"Train MSE={epoch_loss_accum:.6f}, Val MSE={val_loss:.6f}")

# ----------------------------------------------------------------------------
# 11) Test Evaluation
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
# 12) Plot Loss Curves
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
plt.savefig(os.path.join(current_dir, f"{__file__}_train_val_log_mse_and_lr_{base_lr}_{batch_size}_{num_epochs}.png"), dpi=300)
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
# Inside the autoregressive evaluation loop:
with torch.no_grad():
    for t in range(context_window, T_test):
        # Reshape the current window from [1, context_window, 5, 150, 300] to [1, context_window*5, 150, 300]
        window_input = window.reshape(1, context_window * 5, target_h, target_w)
        pred = model(window_input)  # shape: (1, 1, 150, 300)
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
        # Update window: remove the oldest timestep and append new_frame (after reshaping to match 5D window)
        # First, reshape new_frame to (1, 1, 5, 150, 300)
        new_frame = new_frame.unsqueeze(1)
        window = torch.cat([window[:, 1:], new_frame], dim=1)


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
# 13) Plot Predictions (Optional)
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
    fig.savefig(os.path.join(current_dir, f"{__file__}_test_sample_{i}_{base_lr}_{batch_size}_{num_epochs}.png"), dpi=300)
    plt.close(fig)

print("Training, validation, and test evaluation complete.")
print("Final Train MSE:", train_mses[-1])
print("Final Val MSE:", val_mses[-1])
print("Final Test MSE (final):", test_mse_final)
print(f"{N_test_samples} PNG files (one for each test sample) have been saved.")
