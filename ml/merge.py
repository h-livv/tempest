import numpy as np
import random
import os

# 1. Paths to your generated data
paths = [
    '../pipeline_results/advection_rk4_upwind_advec_gauss_periodic_N5000_dt0.005_dx0.05/spatial_data.npz',
    '../pipeline_results/advection_rk4_upwind_advec_square_periodic_N5000_dt0.005_dx0.05/spatial_data.npz'
]

# 2. Lists to hold the data
all_tensors = []
for p in paths:
    data = np.load(p)
    all_tensors.append(data['ml_tensor_data'])

random.shuffle(all_tensors)

# 3. Stack them together (Concatenate along the time axis)
# If Shape A is (5000, 5000) and Shape B is (5000, 5000)
# The result will be (10000, 5000)
master_tensor = np.concatenate(all_tensors, axis=0)

# 4. Save the new master file
np.savez('master_dataset.npz', ml_tensor_data=master_tensor)

print(f"Done! Created master_dataset.npz with shape {master_tensor.shape}")