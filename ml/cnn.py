import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs("output_plots", exist_ok=True)

# --- 1. Data Pipeline ---
file_path = '../pipeline_results/advection_rk4_upwind_advec_square_periodic_N5000_dt0.005_dx0.05/spatial_data.npz'
data = np.load(file_path)
raw_matrix = data['ml_tensor_data'] 

# Pin memory for faster GPU transfer
tensor_matrix = torch.tensor(raw_matrix, dtype=torch.float32).pin_memory().to(device, non_blocking=True)
y_train_np = raw_matrix[1:, :] 

# --- 2. Physics Constraints (Dynamic Energy Conservation) ---
def physics_loss_energy_conservation(input_state, predicted_state):
    # Conserve energy between step T and T+1
    input_energy = torch.mean(input_state**2, dim=1)
    predicted_energy = torch.mean(predicted_state**2, dim=1)
    return torch.mean((predicted_energy - input_energy)**2)

# --- 3. Architecture ---
class ConvSurrogate(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        # Purely linear convolution
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2, padding_mode='circular'),

            nn.Conv1d(16, 16, kernel_size=5, padding=2, padding_mode='circular'),

            nn.Conv1d(16, 1, kernel_size=5, padding=2, padding_mode='circular')
        )

    def forward(self, x):
        # Ensure dimensions match [Batch, Channel, Width]
        return self.network(x.unsqueeze(1)).squeeze(1)

model = ConvSurrogate(tensor_matrix.shape[1]).to(device)

# --- 4. Training Configuration ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
loss_fn = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')

lambda_physics = 0.1  
epochs = 500
unroll_steps = 2
batch_size = 64
max_start_index = tensor_matrix.shape[0] - unroll_steps

print(f"Starting Robust Training...")

# --- 5. Training Loop ---
for epoch in range(epochs):
    optimizer.zero_grad()
    batch_indices = torch.randint(0, max_start_index, (batch_size,), device=device)
    current_lambda = 0.0 if epoch < 200 else 0.1
    
    with torch.amp.autocast('cuda'):
        current_predictions = tensor_matrix[batch_indices, :]
        total_data_loss = 0
        total_physics_loss = 0
        
        for step in range(1, unroll_steps + 1):
            last_state = current_predictions
            current_predictions = model(current_predictions)
            target_states = tensor_matrix[batch_indices + step, :]
            
            total_data_loss += loss_fn(current_predictions, target_states)
            total_physics_loss += physics_loss_energy_conservation(last_state, current_predictions)

        final_loss = (total_data_loss / unroll_steps) + (current_lambda * total_physics_loss)
    
    # Safety: Scale, Clip, and Step
    scaler.scale(final_loss).backward()
    scaler.unscale_(optimizer) 
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
    scaler.step(optimizer)
    scaler.update()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3} | Total Loss: {final_loss.item():.6f}")

# --- 6. Rollout ---
model.eval()
current_state = tensor_matrix[0].unsqueeze(0) 
ml_history = [current_state.squeeze().cpu().detach().numpy()]

with torch.no_grad():
    for _ in range(100):
        current_state = model(current_state)
        ml_history.append(current_state.squeeze().cpu().detach().numpy())

plt.figure(figsize=(10, 5))
plt.plot(y_train_np[99], label="True Physics", color="blue", alpha=0.6)
plt.plot(ml_history[100], label="ML Rollout", linestyle="--", color="red")
plt.legend()
plt.grid(True)
plt.savefig("output_plots/CNN/final_rollout.png")
print("Saved clean plot.")