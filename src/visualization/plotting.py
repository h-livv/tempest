import matplotlib
matplotlib.use("Qt5Agg")

import pandas as pd

import matplotlib.pyplot as plt

# Load your CSV data (adjust filename as needed)
df = pd.read_csv('pipeline_results/wave_validation/euler_vs_leap_vs_rk4.csv')

# Set up a publication-style plot
plt.figure(figsize=(8, 5))

# Plot the lines
plt.plot(df['time'], df['l2_error_rk4'], label='RK4', color='blue', linewidth=1.5)
plt.plot(df['time'], df['l2_error_euler'], label='Euler', color='darkorange', linewidth=1.5)
plt.plot(df['time'], df['l2_error_leapfrog'], label='Leapfrog', color='red', linewidth=1.5)

# Formatting
plt.title('Temporal Evolution of L2 Error', fontsize=14, pad=15)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('L2 Error Norm', fontsize=12)

# Force the y-axis to start at 0, since L2 norms cannot be negative
plt.ylim(bottom=0)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', frameon=True, shadow=True)

# Tight layout prevents axis labels from getting cut off
plt.tight_layout()

# Save as a high-res PNG or PDF
plt.show()