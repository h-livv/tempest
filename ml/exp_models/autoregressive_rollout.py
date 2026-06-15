import numpy as np
import torch
import torch.nn as nn

# 1. Load the data
data = np.load('../pipeline_results/advection_rk4_upwind_advec_gauss_periodic_N5000_dt0.005_dx0.05/spatial_data.npz')
raw_matrix = data['ml_tensor_data']  # Shape: (Total_Time_Steps, Grid_Size_N)

# 2. Create Inputs (T) and Targets (T+1)
x_train_np = raw_matrix[:-1, :] # Everything except the last step
y_train_np = raw_matrix[1:, :]  # Everything except the first step

# 3. Convert to PyTorch Tensors
x_train = torch.tensor(x_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)

def physics_loss_energy_conservation(predicted_state, initial_energy):

    # Calculate the energy of the predicted state
    predicted_energy = torch.mean(predicted_state**2, dim=1)
    
    # Punish any deviation from the starting energy
    conservation_error = torch.mean((predicted_energy - initial_energy)**2)
    
    return conservation_error

class Surrogate(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        # nn.Sequential stacks layers. Standard dense matrix multiplication.
        self.network = nn.Sequential(
            nn.Linear(grid_size, 256),
            nn.ReLU(),  # Non-linear activation function
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, grid_size) # Output must match the grid size
        )

    def forward(self, x):
        return self.network(x)

# Initialize the model
grid_size = x_train.shape[1]
model = Surrogate(grid_size)

# The Optimizer controls how the weights update

# The Loss Function measures error
loss_fn = nn.MSELoss()

# Calculate the true initial mass before training starts
initial_mass = torch.mean(x_train[0])
initial_energy = torch.mean(x_train[0]**2)

# Physics weight
lambda_physics = 0.1  

epochs = 500

for epoch in range(epochs):
    # Predict (Forward Pass)
    predictions = model(x_train)
    
    # Measure Data Error
    loss_data = loss_fn(predictions, y_train)
    
    # Measure Physics Error
    loss_physics = physics_loss_energy_conservation(predictions, initial_energy)
    
    # Combine them for the final backpropagation
    total_loss = loss_data + (lambda_physics * loss_physics)
    
    # Backpropagate using the intertwined loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Data Loss: {loss_data.item():.6f} | Physics Loss: {loss_physics.item():.6f}")



import matplotlib.pyplot as plt

model.eval()

# How many steps into the future we want to simulate
rollout_steps = 500 

# Start with the initial condition at T=0
current_state = x_train[0].unsqueeze(0) 

# History storage
ml_history = [current_state.squeeze().detach().numpy()]

print(f"Starting {rollout_steps}-step autoregressive rollout...")

with torch.no_grad():
    for step in range(rollout_steps):
        #  Predict the next step
        next_state = model(current_state)
        
        # Save the prediction
        ml_history.append(next_state.squeeze().detach().numpy())
        
        # Feed the output back in as the new input
        current_state = next_state

print("Generating visualization...")


# Comparing the model's 100th guess to the true physical state at T=100
step_to_plot = rollout_steps 

ml_final_state = ml_history[step_to_plot]
true_final_state = y_train_np[step_to_plot - 1] # y_train is shifted by 1

plt.figure(figsize=(10, 5))
plt.title(f"Autoregressive Rollout: T={step_to_plot}")

# Plot the True Physics
plt.plot(true_final_state, label=f"True Physics (T={step_to_plot})", color="blue", alpha=0.6, linewidth=2)

# Plot the ML's compounded guess
plt.plot(ml_final_state, label=f"ML Rollout (T={step_to_plot})", linestyle="--", color="red", linewidth=2)

plt.legend()
plt.grid(True)
plt.savefig(f"output_plots/phy_loss_rollout_epoch500_T{step_to_plot}.png", bbox_inches='tight')
print(f"Saved rollout plot as 'phy_loss_rollout_epoch500_T{step_to_plot}.png'")