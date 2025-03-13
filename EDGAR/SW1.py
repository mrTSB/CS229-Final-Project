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
num_molecules = 5    # CH4, CO2, CO2bio, GWP, N2O
num_time = 288       # 12 months * 24 years
target_h, target_w = 75, 150   # target resolution
output_channels = 5  # number of molecules

# Extract emissions data (data type index 0 -> "emi")
data_emissions = kernelized_tensor[:, 0, :, :, :]

# Rearrange dimensions: from (molecules, time, lat, lon) to (time, molecules, lat, lon)
emissions_data = np.transpose(data_emissions, (1, 0, 2, 3)).astype(np.float32)
# Rearrange to (time, height, width, channels)
Y_full = np.transpose(emissions_data, (0, 2, 3, 1))  # Shape: (288, 600, 1200, 5)

# Downsample from 600x1200 to 75x150 via block averaging.
# 600/75 = 8 and 1200/150 = 8.
Y_low = np.empty((num_time, target_h, target_w, output_channels), dtype=np.float32)
for i in range(num_time):
    Y_low[i] = Y_full[i].reshape(target_h, 8, target_w, 8, output_channels).mean(axis=(1, 3))

# Apply log1p transformation
Y_low_log = np.log1p(Y_low)  # Shape: (288, 75, 150, 5)

# ---------------------------------------
# Build Rolling Windows for Training
# ---------------------------------------
context_window = 36  # number of time steps used as input
forecast_horizon = 1 # predict the next step

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
# 90/10 Split: Train and Test (with overlap for autoregressive forecast)
# ---------------------------------------
# For autoregressive evaluation, we want the test set to include a full seed.
# Let train_size = int(0.9*T) and then define test set as from (train_size - context_window) to end.
T_total = Y_low_log.shape[0]   # 288
train_size = int(0.9 * T_total)  # ~259 time steps for training
Y_train_raw = Y_low_log[:train_size]
# Overlap: include the last context_window steps of the training portion in the test set seed.
Y_test_raw = Y_low_log[train_size - context_window:]  # Test sequence length = T_total - (train_size - context_window)

# Create training sequences
X_train, Y_train = create_sequences(Y_train_raw, context_window)
print("X_train shape:", X_train.shape, "Y_train shape:", Y_train.shape)
print("Y_test (raw test sequence) shape:", Y_test_raw.shape)

# ---------------------------------------
# Normalize the Data (using Train Set Stats)
# ---------------------------------------
Y_train_mean = np.mean(Y_train, dtype=np.float32)
Y_train_std  = np.std(Y_train, dtype=np.float32)
X_train_norm = (X_train - Y_train_mean) / Y_train_std
Y_train_norm = (Y_train - Y_train_mean) / Y_train_std
# For autoregressive evaluation, normalize entire test sequence.
Y_test_seq_norm = (Y_test_raw - Y_train_mean) / Y_train_std

# For training, flatten each time slice but keep the sequence dimension.
def flatten_time_slice(X):
    b, t, h, w, c = X.shape
    return X.reshape(b, t, h * w * c)

X_train_norm_flat = flatten_time_slice(X_train_norm)
# For outputs (single time slice), flatten completely.
def flatten_outputs(Y):
    b, h, w, c = Y.shape
    return Y.reshape(b, h * w * c)
Y_train_norm_flat = flatten_outputs(Y_train_norm)

# Convert training arrays to PyTorch tensors.
X_train_tensor = torch.tensor(X_train_norm_flat, dtype=torch.float32, device=device)
Y_train_tensor = torch.tensor(Y_train_norm_flat, dtype=torch.float32, device=device)
print("X_train_tensor shape:", X_train_tensor.shape)  # (num_train_samples, 36, 56250)
print("Y_train_tensor shape:", Y_train_tensor.shape)  # (num_train_samples, 56250)

# ---------------------------------------
# Define the LSTM Model with Xavier Initialization
# ---------------------------------------
# input_dim = 75 * 150 * 5 = 56250
input_dim = target_h * target_w * output_channels
hidden_dim = 128
num_layers = 2
output_dim = input_dim

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
model.fc.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
print(model)

# ---------------------------------------
# Training Loop (Storing only MSE Values)
# ---------------------------------------
num_epochs = 200
train_mses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()
    train_mses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train MSE (norm) = {loss.item():.6f}")

# ---------------------------------------
# Autoregressive Test Evaluation
# ---------------------------------------
def autoregressive_forecast(model, seed, forecast_length):
    """
    Given a seed of shape (context_window, H, W, C) in normalized domain,
    predict forecast_length steps autoregressively.
    Returns an array of shape (forecast_length, H, W, C) in normalized domain.
    """
    model.eval()
    current_window = seed.copy()  # shape: (context_window, H, W, C)
    predictions = []
    for _ in range(forecast_length):
        # Flatten each time slice in the current window (keeping sequence dimension)
        input_tensor = torch.tensor(current_window.reshape(1, context_window, -1),
                                    dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(input_tensor)  # shape: (1, output_dim)
        pred_np = pred.cpu().numpy().reshape(target_h, target_w, output_channels)
        predictions.append(pred_np)
        # Update window: remove the oldest time step and append the new prediction
        current_window = np.concatenate([current_window[1:], pred_np[np.newaxis, ...]], axis=0)
    return np.array(predictions)  # shape: (forecast_length, H, W, C)

# For autoregressive forecasting, the test sequence is Y_test_seq_norm (shape: (T_test,75,150,5))
T_test = Y_test_seq_norm.shape[0]  # Should be 65 in this setting (since 259 - 36 = 223; 288 - 223 = 65)
forecast_length = T_test - context_window  # forecast_length = 65 - 36 = 29
seed = Y_test_seq_norm[:context_window]  # initial seed of shape (36,75,150,5)
preds_norm = autoregressive_forecast(model, seed, forecast_length)  # (29,75,150,5)
# Ground truth: the remaining test sequence (starting from context_window)
gt_norm = Y_test_seq_norm[context_window:]  # (29,75,150,5)

# Inverse normalization
preds_log = preds_norm * Y_train_std + Y_train_mean
gt_log = gt_norm * Y_train_std + Y_train_mean

# Inverse log1p transformation to obtain final emission values
Y_test_pred_final = np.expm1(preds_log)
Y_test_final = np.expm1(gt_log)

mse = mean_squared_error(Y_test_final.flatten(), Y_test_pred_final.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_final.flatten(), Y_test_pred_final.flatten())
r2 = r2_score(Y_test_final.flatten(), Y_test_pred_final.flatten())
corr = np.corrcoef(Y_test_final.flatten(), Y_test_pred_final.flatten())[0, 1]
print("Autoregressive Test MSE:", mse)
print("Autoregressive Test RMSE:", rmse)
print("Autoregressive Test MAE:", mae)
print("Autoregressive Test R^2:", r2)
print("Autoregressive Test Pearson correlation coefficient:", corr)

# ---------------------------------------
# Plot and Save Log of Train MSE (Skipping Early Epochs)
# ---------------------------------------
start_plot_epoch = 10
plot_epochs = range(start_plot_epoch, num_epochs + 1)
train_mses_plot = train_mses[start_plot_epoch - 1:]
plt.figure()
plt.plot(plot_epochs, np.log(train_mses_plot), label='Log Train MSE (norm)')
plt.xlabel('Epoch')
plt.ylabel('Log MSE (Normalized)')
plt.title(f'Log Train MSE (Epoch {start_plot_epoch} onward)')
plt.legend()
plt.savefig(os.path.join(current_dir, "train_log_mse.png"), dpi=300)
plt.show()

# ---------------------------------------
# Helper Function to Tile Images into a Mosaic
# ---------------------------------------
def tile_images(images, ncols):
    """
    images: array of shape (N, H, W) containing grayscale images.
    ncols: number of columns in the mosaic.
    Returns a tiled image array.
    """
    N, H, W = images.shape
    nrows = math.ceil(N / ncols)
    mosaic = np.zeros((nrows * H, ncols * W))
    for idx in range(N):
        r = idx // ncols
        c = idx % ncols
        mosaic[r*H:(r+1)*H, c*W:(c+1)*W] = images[idx]
    return mosaic

# ---------------------------------------
# Create and Save Mosaic PNGs for All Molecules Over All Test Time Steps
# ---------------------------------------
# Y_test_final and Y_test_pred_final have shape (forecast_length, 75, 150, 5)
# We'll create one mosaic for ground truth and one for predictions for each molecule.
ncols = 6  # number of columns in mosaic (adjust as needed)
forecast_length = Y_test_final.shape[0]

for m in range(output_channels):
    # Extract ground truth and prediction images for molecule m over all forecast steps.
    gt_images = np.array([Y_test_final[t, :, :, m] for t in range(forecast_length)])
    pred_images = np.array([Y_test_pred_final[t, :, :, m] for t in range(forecast_length)])
    # Tile the images.
    mosaic_gt = tile_images(gt_images, ncols)
    mosaic_pred = tile_images(pred_images, ncols)
    
    # Plot and save ground truth mosaic.
    plt.figure(figsize=(12, 8))
    norm = LogNorm(vmin=mosaic_gt.min() if mosaic_gt.min() > 0 else 1e-6, vmax=mosaic_gt.max())
    plt.imshow(mosaic_gt, cmap='viridis', norm=norm)
    plt.colorbar(label="Log-Scaled Emissions")
    plt.title(f"Ground Truth Emissions for Molecule {m} (Autoregressive Test Forecast)")
    plt.xlabel("Grid Columns")
    plt.ylabel("Grid Rows")
    plt.savefig(os.path.join(current_dir, f"test_gt_molecule_{m}.png"), dpi=300)
    plt.close()
    
    # Plot and save prediction mosaic.
    plt.figure(figsize=(12, 8))
    norm = LogNorm(vmin=mosaic_pred.min() if mosaic_pred.min() > 0 else 1e-6, vmax=mosaic_pred.max())
    plt.imshow(mosaic_pred, cmap='viridis', norm=norm)
    plt.colorbar(label="Log-Scaled Emissions")
    plt.title(f"Predicted Emissions for Molecule {m} (Autoregressive Test Forecast)")
    plt.xlabel("Grid Columns")
    plt.ylabel("Grid Rows")
    plt.savefig(os.path.join(current_dir, f"test_pred_molecule_{m}.png"), dpi=300)
    plt.close()

print("Mosaic PNGs for all 5 molecules have been saved.")
