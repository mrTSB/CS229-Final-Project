import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader, TensorDataset

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
molecule_types = ["CH4", "CO2", "CO2bio", "GWP", "N2O"]
data_types = ["emi", "flx"]
super_tensor_shape = (5, 2, 288, 600, 1200)
kernelized_tensor = np.memmap(super_tensor_file, dtype="float32", mode="r", shape=super_tensor_shape)
print("Loaded data shape:", kernelized_tensor.shape)

# ---------------------------------------
# Data Preprocessing (Downsample to 75 x 150)
# ---------------------------------------
num_molecules = 5     # 5 molecules
num_time = 288        # 288 timesteps
target_h, target_w = 75, 150   # target resolution
output_channels = 5   # 5 channels

# Extract emissions data (using "emi", index 0)
data_emissions = kernelized_tensor[:, 0, :, :, :]

# Rearrange dimensions from (molecules, time, lat, lon) to (time, molecules, lat, lon)
emissions_data = np.transpose(data_emissions, (1, 0, 2, 3)).astype(np.float32)
# Rearrange to (time, H, W, channels); shape: (288, 600, 1200, 5)
Y_full = np.transpose(emissions_data, (0, 2, 3, 1))

# Downsample from 600x1200 to 75x150 via block averaging.
# (600/75 = 8 and 1200/150 = 8)
Y_low = np.empty((num_time, target_h, target_w, output_channels), dtype=np.float32)
for i in range(num_time):
    Y_low[i] = Y_full[i].reshape(target_h, 8, target_w, 8, output_channels).mean(axis=(1,3))

# Apply log1p transformation
Y_low_log = np.log1p(Y_low)  # shape: (288, 75, 150, 5)

# ---------------------------------------
# Build Rolling Windows for Sequences
# ---------------------------------------
# Use a context window of 24 timesteps and forecast horizon 1.
context_window = 24  
forecast_horizon = 1

def create_sequences(data, window_size):
    T = data.shape[0]
    X_list, y_list = [], []
    for i in range(T - window_size - forecast_horizon + 1):
        X_seq = data[i : i + window_size]
        y_seq = data[i + window_size : i + window_size + forecast_horizon]
        X_list.append(X_seq)
        y_list.append(y_seq[0])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# ---------------------------------------
# 80/10/10 Split: Train, Validation, Test
# ---------------------------------------
T_total = Y_low_log.shape[0]  # 288
train_size = int(0.8 * T_total)       # ~230 timesteps
val_size = int(0.1 * T_total)         # ~28 timesteps
test_size = T_total - train_size - val_size  # ~30 timesteps

Y_train_raw = Y_low_log[:train_size]         # shape: (~230, 75, 150, 5)
Y_val_raw = Y_low_log[train_size:train_size+val_size]  # shape: (~28, 75, 150, 5)
Y_test_raw = Y_low_log[train_size+val_size:]   # shape: (~30, 75, 150, 5)

# For training, create sequences from the training set.
X_train, Y_train_seq = create_sequences(Y_train_raw, context_window)
# Use all 5 channels as input; target is the 5th channel.
X_train_raw = X_train               # shape: (N_train, 24, 75, 150, 5)
Y_train_raw = Y_train_seq[..., 4:]    # shape: (N_train, 75, 150, 1)

# For validation, create sequences from validation set.
X_val, Y_val_seq = create_sequences(Y_val_raw, context_window)
X_val_raw = X_val                   # shape: (N_val, 24, 75, 150, 5)
Y_val_raw = Y_val_seq[..., 4:]       # shape: (N_val, 75, 150, 1)

# For test, create sequences from test set.
X_test, Y_test_seq = create_sequences(Y_test_raw, context_window)
X_test_raw = X_test                 # shape: (N_test, 24, 75, 150, 5)
Y_test_raw = Y_test_seq[..., 4:]     # shape: (N_test, 75, 150, 1)

print("Training sequences: X_train_raw:", X_train_raw.shape, "Y_train_raw:", Y_train_raw.shape)
print("Validation sequences: X_val_raw:", X_val_raw.shape, "Y_val_raw:", Y_val_raw.shape)
print("Test sequences: X_test_raw:", X_test_raw.shape, "Y_test_raw:", Y_test_raw.shape)

# ---------------------------------------
# Normalize Data (using Training Set Stats)
# ---------------------------------------
# Compute statistics over training data.
X_train_mean = np.mean(X_train_raw, axis=(0,1,2,3), keepdims=True)
X_train_std  = np.std(X_train_raw, axis=(0,1,2,3), keepdims=True)
Y_train_mean = np.mean(Y_train_raw, axis=(0,1,2,3), keepdims=True)
Y_train_std  = np.std(Y_train_raw, axis=(0,1,2,3), keepdims=True)

X_train_norm = (X_train_raw - X_train_mean) / X_train_std
Y_train_norm = (Y_train_raw - Y_train_mean) / Y_train_std

X_val_norm = (X_val_raw - X_train_mean) / X_train_std
Y_val_norm = (Y_val_raw - Y_train_mean) / Y_train_std

X_test_norm = (X_test_raw - X_train_mean) / X_train_std
Y_test_norm = (Y_test_raw - Y_train_mean) / Y_train_std

# ---------------------------------------
# Convert Training and Validation Data to Tensors
# ---------------------------------------
# For ConvLSTM, input shape: (batch, seq_len, channels, H, W)
X_train_norm = X_train_norm.transpose(0, 1, 4, 2, 3)  # becomes (N_train, 24, 5, 75, 150)
Y_train_norm = Y_train_norm.transpose(0, 3, 1, 2)       # becomes (N_train, 1, 75, 150)
train_dataset = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32, device=device),
                                torch.tensor(Y_train_norm, dtype=torch.float32, device=device))
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# For validation and test, use the sequences as they are.
X_val_tensor = torch.tensor(X_val_norm.transpose(0, 1, 4, 2, 3), dtype=torch.float32, device=device)   # (N_val, 24, 5, 75, 150)
Y_val_tensor = torch.tensor(Y_val_norm.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)        # (N_val, 1, 75, 150)
X_test_tensor = torch.tensor(X_test_norm.transpose(0, 1, 4, 2, 3), dtype=torch.float32, device=device)     # (N_test, 24, 5, 75, 150)
Y_test_tensor = torch.tensor(Y_test_norm.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)      # (N_test, 1, 75, 150)

# ---------------------------------------
# Define the ConvLSTM Model with Regularization (Dropout + Weight Decay)
# ---------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding)
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
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
            h, c = self.cell(input_tensor[:, t, ...], (h, c))
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
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()
print(model)

# ---------------------------------------
# Training Loop (Using DataLoader)
# ---------------------------------------
num_epochs = 200
train_mses = []
val_mses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)  # (batch, 1, 75, 150)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= len(train_dataset)
    train_mses.append(epoch_loss)
    
    # Evaluate on validation set (normal testing)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)  # (N_val, 1, 75, 150)
        val_loss = criterion(val_outputs, Y_val_tensor)
    val_mses.append(val_loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train MSE = {epoch_loss:.6f}, Val MSE = {val_loss.item():.6f}")

# ---------------------------------------
# Test Evaluation (Normal Testing)
# ---------------------------------------
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)  # (N_test, 1, 75, 150)
test_loss = criterion(test_outputs, Y_test_tensor).item()
print("Test MSE (normalized):", test_loss)

def unflatten_outputs(Y, h=target_h, w=target_w, c=1):
    b = Y.shape[0]
    return Y.reshape(b, h, w, c)

Y_test_pred = test_outputs.cpu().numpy().reshape(-1, target_h, target_w, 1)
Y_test_true = Y_test_tensor.cpu().numpy().reshape(-1, target_h, target_w, 1)
# Inverse normalization:
Y_test_pred_log = Y_test_pred * Y_train_std + Y_train_mean
Y_test_true_log = Y_test_true * Y_train_std + Y_train_mean
# Inverse log1p transformation:
Y_test_pred_final = np.expm1(Y_test_pred_log)
Y_test_true_final = np.expm1(Y_test_true_log)
test_mse_final = mean_squared_error(Y_test_true_final.flatten(), Y_test_pred_final.flatten())
print("Test MSE (final):", test_mse_final)

# ---------------------------------------
# Plot and Save Log of Train and Validation MSE (Skipping Early Epochs)
# ---------------------------------------
start_plot_epoch = 10
plot_epochs = range(start_plot_epoch, num_epochs + 1)
train_mses_plot = train_mses[start_plot_epoch - 1:]
val_mses_plot = val_mses[start_plot_epoch - 1:]
plt.figure()
plt.plot(plot_epochs, np.log(train_mses_plot), label='Log Train MSE')
plt.plot(plot_epochs, np.log(val_mses_plot), label='Log Val MSE')
plt.xlabel('Epoch')
plt.ylabel('Log MSE')
plt.title(f'Log MSE (Epoch {start_plot_epoch} onward)')
plt.legend()
plt.savefig(os.path.join(current_dir, "train_val_log_mse.png"), dpi=300)
plt.show()

# ---------------------------------------
# Plot Specific Emission Comparisons for Selected Test Samples
# ---------------------------------------
def plot_emission_data(data, title, extent, cmap='viridis'):
    """Plot emission data with log scaling and proper orientation."""
    vmin = data.min() if data.min() > 0 else 1e-6
    norm = LogNorm(vmin=vmin, vmax=data.max())
    plt.figure(figsize=(6,6))
    plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, norm=norm)
    plt.colorbar(label="Log-Scaled Emissions")
    plt.title(title)
    plt.xlabel("Longitude Grid Points")
    plt.ylabel("Latitude Grid Points")
    plt.gca().invert_yaxis()
    return plt.gcf()

def plot_specific_emission_comparison(tensor_gt, tensor_pred, time_index, extent, cmap='viridis'):
    """
    Plot ground truth and predicted emission for the 5th molecule at a given time index.
    Args:
        tensor_gt (np.array): Ground truth tensor of shape (T, H, W, 1) in final domain.
        tensor_pred (np.array): Predicted tensor of shape (T, H, W, 1) in final domain.
        time_index (int): Index (within test set) to plot.
        extent (list): Plot extent, e.g., [0, target_w, 0, target_h].
        cmap (str): Colormap.
    """
    gt_img = tensor_gt[time_index, :, :, 0]
    pred_img = tensor_pred[time_index, :, :, 0]
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    norm_gt = LogNorm(vmin=gt_img.min() if gt_img.min() > 0 else 1e-6, vmax=gt_img.max())
    norm_pred = LogNorm(vmin=pred_img.min() if pred_img.min() > 0 else 1e-6, vmax=pred_img.max())
    im0 = axs[0].imshow(gt_img, cmap=cmap, interpolation='nearest', extent=extent, norm=norm_gt)
    fig.colorbar(im0, ax=axs[0], label="Log-Scaled Emissions")
    axs[0].set_title(f"Ground Truth at Sample {time_index}")
    axs[0].set_xlabel("Longitude Grid Points")
    axs[0].set_ylabel("Latitude Grid Points")
    axs[0].invert_yaxis()
    
    im1 = axs[1].imshow(pred_img, cmap=cmap, interpolation='nearest', extent=extent, norm=norm_pred)
    fig.colorbar(im1, ax=axs[1], label="Log-Scaled Emissions")
    axs[1].set_title(f"Prediction at Sample {time_index}")
    axs[1].set_xlabel("Longitude Grid Points")
    axs[1].set_ylabel("Latitude Grid Points")
    axs[1].invert_yaxis()
    
    plt.tight_layout()
    return fig

# Select 5 evenly spaced indices from the test sequences.
N_test = Y_test_true_final.shape[0]
selected_indices = np.linspace(0, N_test - 1, 5, dtype=int)
extent = [0, target_w, 0, target_h]

for idx in selected_indices:
    fig = plot_specific_emission_comparison(Y_test_true_final, Y_test_pred_final, idx, extent, cmap='viridis')
    fig.savefig(os.path.join(current_dir, f"test_sample_{idx}.png"), dpi=300)
    plt.close(fig)

print("Training, validation, and test evaluation complete.")
print("Train MSE:", train_mses[-1])
print("Val MSE:", val_mses[-1])
print("Test MSE (final):", test_mse_final)
print("Five PNG files (each with ground truth and prediction) have been saved for selected test samples.")
