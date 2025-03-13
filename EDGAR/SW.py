import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Define molecule types and data types
molecule_types = ["CH4", "CO2", "CO2bio", "GWP", "N2O"]
data_types = ["emi", "flx"]

# Load large tensor from disk (Memory-Mapped)
super_tensor_shape = (5, 2, 288, 600, 1200)
kernelized_tensor = np.memmap(
    super_tensor_file, dtype="float32", mode="r", shape=super_tensor_shape
)

print("Loaded data shape:", kernelized_tensor.shape)

# ---------------------------------------
# Data Preprocessing
# ---------------------------------------
num_molecules = 5   # CH4, CO2, CO2bio, GWP, N2O
num_time = 288      # 12 months * 24 years
lat_full = 600
lon_full = 1200

low_res_h, low_res_w = 75, 150   # Target resolution
output_channels = 5              # Number of molecules

# Extract emissions data (data type index 0 -> "emi")
data_emissions = kernelized_tensor[:, 0, :, :, :]

# Rearrange dimensions: (molecules, time, lat, lon) -> (time, molecules, lat, lon)
emissions_data = np.transpose(data_emissions, (1, 0, 2, 3)).astype(np.float32)
# shape: (288, 5, 600, 1200)

# Downsample from (600, 1200) to (75, 150) using block averaging
Y_full = np.transpose(emissions_data, (0, 2, 3, 1))  # shape: (288, 600, 1200, 5)
Y_low = np.empty((num_time, low_res_h, low_res_w, output_channels), dtype=np.float32)

for i in range(num_time):
    # Reshape to (75, 8, 150, 8, 5) and average over axes 1 and 3
    Y_low[i] = Y_full[i].reshape(low_res_h, 8, low_res_w, 8, output_channels).mean(axis=(1, 3))

# Apply log1p transformation
Y_low_log = np.log1p(Y_low)  # shape: (288, 75, 150, 5)

# ---------------------------------------
# Build rolling windows for time series
# ---------------------------------------
context_window = 36  # Number of time steps used as input
forecast_horizon = 1 # Predict the next step

def create_sequences(data, window_size):
    """
    data: shape (T, H, W, C)
    window_size: int
    Returns:
        X: shape (num_samples, window_size, H, W, C)
        y: shape (num_samples, H, W, C)
    """
    T = data.shape[0]
    X_list, y_list = [], []
    for i in range(T - window_size - forecast_horizon + 1):
        X_seq = data[i : i + window_size]  
        y_seq = data[i + window_size : i + window_size + forecast_horizon]
        X_list.append(X_seq)
        y_list.append(y_seq[0])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# ---------------------------------------
# 60/20/20 Split: Train, Validation, Test
# ---------------------------------------
T = Y_low_log.shape[0]  # 288
train_size = int(0.6 * T)  # ~172
val_size   = int(0.2 * T)  # ~57
train_end  = train_size
val_end    = train_size + val_size

Y_train_raw = Y_low_log[:train_end]
Y_val_raw   = Y_low_log[train_end:val_end]
Y_test_raw  = Y_low_log[val_end:]

X_train, Y_train = create_sequences(Y_train_raw, context_window)
X_val,   Y_val   = create_sequences(Y_val_raw,   context_window)
X_test,  Y_test  = create_sequences(Y_test_raw,  context_window)

print("X_train shape:", X_train.shape, "Y_train shape:", Y_train.shape)
print("X_val   shape:", X_val.shape,   "Y_val   shape:", Y_val.shape)
print("X_test  shape:", X_test.shape,  "Y_test shape:", Y_test.shape)

# ---------------------------------------
# Normalize the Data (using train set stats)
# ---------------------------------------
Y_train_mean = np.mean(Y_train, dtype=np.float32)
Y_train_std  = np.std(Y_train, dtype=np.float32)

X_train_norm = (X_train - Y_train_mean) / Y_train_std
Y_train_norm = (Y_train - Y_train_mean) / Y_train_std

X_val_norm   = (X_val - Y_train_mean)   / Y_train_std
Y_val_norm   = (Y_val - Y_train_mean)   / Y_train_std

X_test_norm  = (X_test - Y_train_mean)  / Y_train_std
Y_test_norm  = (Y_test - Y_train_mean)  / Y_train_std

def flatten_inputs(X):
    b, t, h, w, c = X.shape
    return X.reshape(b, t * h * w * c)

def flatten_outputs(Y):
    b, h, w, c = Y.shape
    return Y.reshape(b, h * w * c)

# Convert arrays to PyTorch tensors
X_train_tensor = torch.tensor(flatten_inputs(X_train_norm), dtype=torch.float32, device=device)
Y_train_tensor = torch.tensor(flatten_outputs(Y_train_norm), dtype=torch.float32, device=device)

X_val_tensor   = torch.tensor(flatten_inputs(X_val_norm),   dtype=torch.float32, device=device)
Y_val_tensor   = torch.tensor(flatten_outputs(Y_val_norm),  dtype=torch.float32, device=device)

X_test_tensor  = torch.tensor(flatten_inputs(X_test_norm),  dtype=torch.float32, device=device)
Y_test_tensor  = torch.tensor(flatten_outputs(Y_test_norm), dtype=torch.float32, device=device)

print("Flattened X_train_tensor shape:", X_train_tensor.shape)
print("Flattened Y_train_tensor shape:", Y_train_tensor.shape)
print("Flattened X_val_tensor shape:", X_val_tensor.shape)
print("Flattened Y_val_tensor shape:", Y_val_tensor.shape)
print("Flattened X_test_tensor shape:", X_test_tensor.shape)
print("Flattened Y_test_tensor shape:", Y_test_tensor.shape)

# ---------------------------------------
# Define the Model with Xavier Initialization
# ---------------------------------------
input_dim = context_window * low_res_h * low_res_w * output_channels
output_dim = low_res_h * low_res_w * output_channels

class MLPModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLPModel(input_dim, output_dim).to(device)

# Xavier init function
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Apply Xavier initialization
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()  # MSELoss returns the MSE (which is our metric)

print(model)

# ---------------------------------------
# Training Loop (storing only MSE values)
# ---------------------------------------
num_epochs = 200

train_mses = []
val_mses = []

for epoch in range(num_epochs):
    # TRAIN
    model.train()
    optimizer.zero_grad()
    
    train_outputs = model(X_train_tensor)
    train_mse = criterion(train_outputs, Y_train_tensor)
    train_mse.backward()
    optimizer.step()
    
    train_mses.append(train_mse.item())
    
    # VALIDATION
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_mse = criterion(val_outputs, Y_val_tensor)
    val_mses.append(val_mse.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train MSE (norm) = {train_mse.item():.6f}, Val MSE (norm) = {val_mse.item():.6f}")

# ---------------------------------------
# Final Evaluation on the Test Set
# ---------------------------------------
model.eval()
with torch.no_grad():
    Y_test_pred_norm_tensor = model(X_test_tensor)

Y_test_pred_norm = Y_test_pred_norm_tensor.cpu().numpy()
Y_test_norm_np    = Y_test_tensor.cpu().numpy()

test_mse_norm = mean_squared_error(Y_test_norm_np.flatten(), Y_test_pred_norm.flatten())
print("Final Test MSE (normalized):", test_mse_norm)

def unflatten_outputs(Y, h=low_res_h, w=low_res_w, c=output_channels):
    b = Y.shape[0]
    return Y.reshape(b, h, w, c)

Y_test_pred_norm_4D = unflatten_outputs(Y_test_pred_norm)
Y_test_norm_4D      = unflatten_outputs(Y_test_norm_np)

Y_test_pred_log = Y_test_pred_norm_4D * Y_train_std + Y_train_mean
Y_test_log      = Y_test_norm_4D      * Y_train_std + Y_train_mean

Y_test_pred = np.expm1(Y_test_pred_log)
Y_test_inv  = np.expm1(Y_test_log)

mse = mean_squared_error(Y_test_inv.flatten(), Y_test_pred.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_inv.flatten(), Y_test_pred.flatten())
r2 = r2_score(Y_test_inv.flatten(), Y_test_pred.flatten())
corr = np.corrcoef(Y_test_inv.flatten(), Y_test_pred.flatten())[0, 1]

print("Final Test MSE:", mse)
print("Final Test RMSE:", rmse)
print("Final Test MAE:", mae)
print("Final Test R^2:", r2)
print("Final Test Pearson correlation coefficient:", corr)

# ---------------------------------------
# Plot and Save Log of Train vs. Validation MSE, Skipping First Few Epochs
# ---------------------------------------
start_plot_epoch = 10  # skip the first 9 epochs in the plot
plot_epochs = range(start_plot_epoch, num_epochs + 1)

# Slice the stored MSE arrays so we skip the earliest epochs
train_mses_plot = train_mses[start_plot_epoch - 1:]
val_mses_plot   = val_mses[start_plot_epoch - 1:]

plt.figure()
plt.plot(plot_epochs, np.log(train_mses_plot), label='Log Train MSE (norm)')
plt.plot(plot_epochs, np.log(val_mses_plot), label='Log Val MSE (norm)')
plt.xlabel('Epoch')
plt.ylabel('Log MSE (Normalized)')
plt.title(f'Log Train vs. Validation MSE (Epoch {start_plot_epoch} onward)')
plt.legend()
plt.savefig(os.path.join(current_dir, "train_val_log_mse.png"), dpi=300)
plt.show()

# ---------------------------------------
# Visualize and Save a Sample Prediction
# ---------------------------------------
sample_idx = 0
molecule_idx = 0

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(Y_test_inv[sample_idx, :, :, molecule_idx], cmap='viridis')
plt.title('Ground Truth (Molecule 0)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(Y_test_pred[sample_idx, :, :, molecule_idx], cmap='viridis')
plt.title('Prediction (Molecule 0)')
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(current_dir, "sample_prediction.png"), dpi=300)
plt.show()
