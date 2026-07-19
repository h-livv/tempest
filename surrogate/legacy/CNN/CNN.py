import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
import datetime
import csv
import time
import sys
import uuid
import glob

# Add parent directory to sys path so we can import modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Determine which dataset to load (Input Data)
if len(sys.argv) < 2:
    print("No dataset directory provided. Searching for the absolute latest simulation...")
    data_base_dirs = ["pipeline_results/runs", "pipeline_results/sweeps"]
    all_runs = []
    
    for base_dir in data_base_dirs:
        if os.path.exists(base_dir):
            for d in os.listdir(base_dir):
                full_path = os.path.join(base_dir, d)
                if os.path.isdir(full_path):
                    all_runs.append(full_path)
                    
    if not all_runs:
        print("Error: Could not automatically find any simulation data in pipeline_results/runs or sweeps")
        sys.exit(1)
        
    # Sort strictly by the folder's modification time (oldest to newest)
    all_runs.sort(key=lambda x: os.path.getmtime(x))
    
    # Grab the last one in the list (the newest)
    data_path = all_runs[-1]
    print(f"✅ Automatically loading chronologically latest dataset: {data_path}")
else:
    data_path = sys.argv[1]

if not os.path.exists(data_path):
    print(f"Error: Could not find {data_path}")
    sys.exit(1)

# Check if sweep or single
npz_files = []
if os.path.isfile(os.path.join(data_path, 'spatial_data.npz')):
    npz_files.append(os.path.join(data_path, 'spatial_data.npz'))
    run_type = "single"
else:
    # Assume it's a sweep folder and search subdirectories
    npz_files = glob.glob(os.path.join(data_path, '*', 'spatial_data.npz'))
    run_type = "sweep"

if not npz_files:
    raise ValueError(f"No spatial_data.npz found in {data_path}")

# Load the sweep .npz files into a list of PyTorch tensors with metadata
print(f"Loading {len(npz_files)} dataset(s) into memory...")
dataset_info = []
tensor_list = []
base_shape = None

for file in npz_files:
    data = np.load(file)
    raw_matrix = data['ml_tensor_data']
    
    if base_shape is None:
        base_shape = raw_matrix.shape[1]
    elif raw_matrix.shape[1] != base_shape:
        print(f"⚠️ Skipping {file} due to spatial shape mismatch: {raw_matrix.shape[1]} vs {base_shape}")
        continue
        
    tensor = torch.tensor(raw_matrix, dtype=torch.float32).pin_memory().to(device, non_blocking=True)
    
    # Load config.json to find the specific initial condition name for this file
    ic_name = "unknown_ic"
    config_path = os.path.join(os.path.dirname(file), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg_data = json.load(f)
                ic_name = cfg_data.get("initial_condition", ic_name).replace(" ", "_").lower()
        except Exception:
            pass
            
    # Do not train on the unseen test condition
    if "shifted" not in ic_name:
        # Train on the first 2500 timesteps, but predict up to 10000
        tensor_list.append(tensor[:2501])
    else:
        print(f"Holding out unseen condition from training: {ic_name}")
            
    dataset_info.append({
        "tensor": tensor,
        "ic_name": ic_name,
        "file_path": file
    })

if not tensor_list:
    raise ValueError("No valid datasets loaded after shape filtering.")

print(f"✅ Successfully loaded {len(tensor_list)} valid dataset(s) for training.")

# 2. Define time and UID for the ML Output Run Directory
ist_timezone = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
timestamp = datetime.datetime.now(ist_timezone).strftime("%Y%m%d_%H%M%S")
uid = uuid.uuid4().hex[:4]

# Extract run metadata from simulation config.json files
eq_name = "unknown_eq"
int_name = "unknown_int"
op_name = "unknown_op"
ic_list = []

for file in npz_files:
    config_path = os.path.join(os.path.dirname(file), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg_data = json.load(f)
                eq_name = cfg_data.get("equation", eq_name)
                int_name = cfg_data.get("integrator", int_name)
                op_name = cfg_data.get("operator", op_name)
                ic = cfg_data.get("initial_condition")
                if ic and ic not in ic_list:
                    ic_list.append(ic)
        except Exception:
            pass

clean_eq = str(eq_name).replace(" ", "_").lower()
clean_int = str(int_name).replace(" ", "_").lower()
clean_op = str(op_name).replace(" ", "_").lower()
clean_ic = str("_".join(sorted(ic_list)) if ic_list else "unknown_ic").replace(" ", "_").lower()

run_dir_name = f"run_{clean_eq}_{clean_int}_{clean_op}_{clean_ic}_{timestamp}_{uid}"
run_dir = os.path.join("ml", "CNN", "runs", "sweeps" if run_type == "sweep" else "single", run_dir_name)
plots_dir = os.path.join(run_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# --- 3. Physics Constraints (Dynamic Energy Conservation) ---
def physics_loss_energy_conservation(input_state, predicted_state):
    # Conserve energy between step T and T+1
    input_energy = torch.mean(input_state**2, dim=1)
    predicted_energy = torch.mean(predicted_state**2, dim=1)
    return torch.mean((predicted_energy - input_energy)**2)

# --- 4. Architecture ---
import torch.nn.functional as F

class ConvSurrogate(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        # Asymmetric Upwind Bias Initialization
        initial_weights = torch.zeros(1, 1, 25)
        initial_weights[0, 0, 10] = 0.1  # Bias towards Shift 2
        self.raw_weights = nn.Parameter(initial_weights)
        self.temperature = 1000.0

    def forward(self, x):
        # Softmax forces a Markov stochastic matrix constraint (positive, sum to 1)
        # Temperature Annealing applied during training loop
        w = F.softmax(self.raw_weights * self.temperature, dim=-1)
        
        # Manually apply circular padding
        x_padded = F.pad(x.unsqueeze(1), (12, 12), mode='circular')
        
        # Apply convolution
        return F.conv1d(x_padded, w).squeeze(1)

model = ConvSurrogate(base_shape).to(device)

# --- 5. Training Configuration ---
learning_rate = 5e-3
# Lighter weight decay to prevent over-damping the linear wave
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6) 
loss_fn = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')

lambda_physics = 0.0  # Disabled: Markov matrix mathematically guarantees baseline stability
epochs = 1000
batch_size = 64
seed = 42
test_timesteps = 10000

torch.manual_seed(seed)

hyperparameters = {
    "learning_rate": learning_rate,
    "lambda_physics": lambda_physics,
    "epochs": epochs,
    "batch_size": batch_size,
    "max_unroll_steps": 8,
    "seed": seed,
    "test_timesteps": test_timesteps,
    "training_data_path": data_path,
    "datasets_loaded": len(tensor_list)
}

# SAVE TO RUN_DIR
with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
    json.dump(hyperparameters, f, indent=4)

metrics_file_path = os.path.join(run_dir, "metrics.csv")
with open(metrics_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "data_loss", "physics_loss", "epoch_time"])

print(f"🚀 Starting Robust Training... Outputs will be saved to {run_dir}")

# --- 6. Training Loop ---
for epoch in range(epochs):
    epoch_start_time = time.time()
    optimizer.zero_grad()
    
    # Temperature Annealing Schedule
    if epoch < 500:
        model.temperature = 1.0 * (1000.0 ** (epoch / 500.0))
    else:
        model.temperature = 1000.0
        
    # Curriculum Unroll Schedule
    # Longer Initial Unroll Horizons
    current_unroll = 8
        
    # Update the Batch Sampler: Randomly select ONE dataset
    dataset_idx = torch.randint(0, len(tensor_list), (1,)).item()
    current_tensor = tensor_list[dataset_idx]
    
    max_start_index = current_tensor.shape[0] - current_unroll
    batch_indices = torch.randint(0, max_start_index, (batch_size,), device=device)
    
    # Data-First Schedule: train purely on data for the first 400 epochs
    current_lambda = 0.0 if epoch < 400 else lambda_physics
    
    with torch.amp.autocast('cuda'):
        current_predictions = current_tensor[batch_indices, :]
        total_data_loss = 0
        total_physics_loss = 0
        
        for step in range(1, current_unroll + 1):
            last_state = current_predictions
            current_predictions = model(current_predictions)
            target_states = current_tensor[batch_indices + step, :]
            
            total_data_loss += loss_fn(current_predictions, target_states)
            total_physics_loss += physics_loss_energy_conservation(last_state, current_predictions)

        data_loss_val = total_data_loss / current_unroll
        physics_loss_val = total_physics_loss
        
        final_loss = data_loss_val + (current_lambda * physics_loss_val)
    
    # Safety: Scale, Clip, and Step
    scaler.scale(final_loss).backward()
    scaler.unscale_(optimizer) 
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
    scaler.step(optimizer)
    scaler.update()
    
    epoch_time = time.time() - epoch_start_time
    
    with open(metrics_file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, data_loss_val.item(), physics_loss_val.item(), epoch_time])

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3} | Total Loss: {final_loss.item():.6f}")

# --- 7. Rollout ---
model.eval()
print("🎉 Training Complete!")

# Save the frozen model weights
model_save_path = os.path.join(run_dir, "advec_model.pt")
torch.save(model.state_dict(), model_save_path)
print(f"Saved model weights to {model_save_path}")

print("\n🚀 Beginning Full Sequence Rollout Verification...")
# We will generate a rollout for each loaded dataset/initial condition dynamically
for idx, info in enumerate(dataset_info):
    ref_tensor = info["tensor"]
    ic_name = info["ic_name"]
    
    # Run the rollout for the length of this simulation
    num_timesteps = ref_tensor.shape[0] - 1
    
    current_state = ref_tensor[0].unsqueeze(0)
    ml_history = [current_state.squeeze().cpu().detach().numpy()]
    
    # Compute initial physical energy for hard conservation projection
    initial_energy = torch.mean(ref_tensor[0]**2)
    
    with torch.no_grad():
        for _ in range(num_timesteps):
            current_state = model(current_state)
            
            # Apply soft energy conservation projection (alpha = 0.2 relaxation factor)
            current_energy = torch.mean(current_state**2)
            current_state = current_state * (0.8 + 0.2 * torch.sqrt(initial_energy / (current_energy + 1e-8)))
            
            ml_history.append(current_state.squeeze().cpu().detach().numpy())
            
    # Save comparison rollout data
    # Contains:
    # - 'rollout': ML predictions of shape (num_timesteps + 1, grid_size)
    # - 'numerical': Ground truth numerical solver history of shape (num_timesteps + 1, grid_size)
    save_path = os.path.join(run_dir, f"rollout_{ic_name}.npz")
    np.savez_compressed(
        save_path, 
        rollout=np.array(ml_history), 
        numerical=ref_tensor.cpu().numpy()
    )
    print(f"Saved rollout comparison for {ic_name} to {save_path}")

    # Generate validation plots for multiple timesteps
    for history_idx in [100, 500, 1000, 2000, 5000, 10000]:
        if history_idx <= num_timesteps:
            plt.figure(figsize=(10, 5))
            analytical = np.roll(ref_tensor[0].cpu().numpy(), shift=2 * history_idx)
            plt.plot(analytical, label=f"Analytical (T={history_idx})", color="green", linestyle=":", linewidth=2, alpha=0.8)
            plt.plot(ref_tensor[history_idx].cpu().numpy(), label=f"Numerical (T={history_idx})", color="blue", alpha=0.6)
            plt.plot(ml_history[history_idx], label=f"ML Rollout (T={history_idx})", linestyle="--", color="red")
            plt.legend()
            plt.grid(True)
            plt.title(f"Validation Rollout Comparison ({ic_name}) at T={history_idx}")
            plt.savefig(os.path.join(plots_dir, f"final_rollout_{ic_name}_T{history_idx}.png"))
            plt.close()

# Save info (Registry removed)
print(f"✅ Training complete. All artifacts saved successfully to {run_dir}")