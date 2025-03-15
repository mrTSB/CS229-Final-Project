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

# ---------------------------------------
# Load Data
# ---------------------------------------
current_dir = os.getcwd()
super_tensor_file = os.path.join(current_dir, "kernelized_tensor.npy")

# Define molecule types and data types
molecule_types = ["CH4", "CO2", "CO2bio", "GWP", "N2O"]
data_types = ["emi", "flx"]

# Load the large tensor from disk (Memory-Mapped)
super_tensor_shape = (5, 2, 288, 600, 1200)
kernelized_tensor = np.memmap(super_tensor_file, dtype="float32", mode="r", shape=super_tensor_shape)

print("Loaded data shape:", kernelized_tensor.shape)

# ---------------------------------------
# Data Preprocessing
# ---------------------------------------
num_molecules = 5  # CH4, CO2, CO2bio, GWP, N2O
num_time = 288  # 12 months * 24 years
lat_full = 600
lon_full = 1200

low_res_h, low_res_w = 75, 150  # Target resolution
output_channels = 5  # Number of molecules

# Extract emissions data (data type index 0)
data_emissions = kernelized_tensor[:, 0, :, :, :]

# Rearrange dimensions (time first)
emissions_data = np.transpose(data_emissions, (1, 0, 2, 3)).astype(np.float32)

# Downsample from (600,1200) to (75,150) using block averaging
Y_full = np.transpose(emissions_data, (0, 2, 3, 1))  # Shape: (288, 600, 1200, 5)
Y_low = np.empty((num_time, low_res_h, low_res_w, output_channels), dtype=np.float32)

for i in range(num_time):
    Y_low[i] = Y_full[i].reshape(low_res_h, 8, low_res_w, 8, output_channels).mean(axis=(1, 3))

# Apply log1p transformation
Y_low_log = np.log1p(Y_low)

# Encode time cyclically
months = np.arange(num_time, dtype=np.float32)
month_norm = months / (num_time - 1)
X = np.stack([np.sin(2 * np.pi * month_norm), np.cos(2 * np.pi * month_norm)], axis=1)

# Train/Test split
train_time = 48
test_time = 12
X_train = X[:train_time]
X_test = X[train_time:train_time + test_time]
Y_train = Y_low_log[:train_time]
Y_test = Y_low_log[train_time:train_time + test_time]

# Normalize the targets
Y_train_mean = np.mean(Y_train, dtype=np.float32)
Y_train_std = np.std(Y_train, dtype=np.float32)
Y_train_norm = (Y_train - Y_train_mean) / Y_train_std
Y_test_norm = (Y_test - Y_train_mean) / Y_train_std

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
Y_train_t = torch.tensor(Y_train_norm, dtype=torch.float32, device=device)
Y_test_t = torch.tensor(Y_test_norm, dtype=torch.float32, device=device)

# ---------------------------------------
# Define PyTorch Model (MLP)
# ---------------------------------------
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, low_res_h * low_res_w * output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x.view(-1, low_res_h, low_res_w, output_channels)

# Initialize the model
model = MLPModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print(model)

# ---------------------------------------
# Training Loop
# ---------------------------------------
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, Y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.6f}")

# ---------------------------------------
# Evaluation
# ---------------------------------------
model.eval()
with torch.no_grad():
    Y_pred_norm_t = model(X_test_t)

# Convert back to NumPy
Y_pred_norm = Y_pred_norm_t.cpu().numpy()
Y_test_norm_np = Y_test_t.cpu().numpy()

# Compute test MSE in the normalized space
test_mse_norm = mean_squared_error(Y_test_norm_np.flatten(), Y_pred_norm.flatten())
print("Test MSE (normalized):", test_mse_norm)

# Inverse transformations
Y_pred_log = Y_pred_norm * Y_train_std + Y_train_mean
Y_test_log = Y_test_norm_np * Y_train_std + Y_train_mean

# Invert log1p
Y_pred = np.expm1(Y_pred_log)
Y_test_inv = np.expm1(Y_test_log)

# ---------------------------------------
# Compute Additional Metrics
# ---------------------------------------
mse = mean_squared_error(Y_test_inv.flatten(), Y_pred.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_inv.flatten(), Y_pred.flatten())
r2 = r2_score(Y_test_inv.flatten(), Y_pred.flatten())
corr = np.corrcoef(Y_test_inv.flatten(), Y_pred.flatten())[0, 1]

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R^2:", r2)
print("Pearson correlation coefficient:", corr)

# ---------------------------------------
# Visualize a Sample Prediction
# ---------------------------------------
sample_idx = 0  # first test sample
molecule_idx = 0  # visualize molecule 0

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(Y_test_inv[sample_idx, :, :, molecule_idx], cmap='viridis')
plt.title('Ground Truth (Molecule 0)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(Y_pred[sample_idx, :, :, molecule_idx], cmap='viridis')
plt.title('Prediction (Molecule 0)')
plt.colorbar()

plt.tight_layout()
plt.show()
