from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Locate and read the CSV safely
SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR / 'advection' /'advection_upwind.csv'
df = pd.read_csv(csv_path)

# 2. Create the plot layout
plt.figure(figsize=(10, 5))

# Plot L2 Error
plt.plot(df['time'], df['l2_error'], marker='o', linestyle='-', color='b', label='$L^2$ Error')

# Plot Relative Error (Optional: you can comment this out if you only want L2)
plt.plot(df['time'], df['relative_error'], marker='x', linestyle='--', color='r', label='Relative Error')

# 3. Add formatting
plt.xlabel('Time (s)')
plt.ylabel('Error Magnitude')
plt.title('Advection Simulation Error Analysis')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# 4. Show the plot
plt.tight_layout()
plt.show()