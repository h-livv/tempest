import numpy as np
import pandas as pd

class DataTracker:
    def __init__(self, final_time, dt, record_interval, grid_size):
        total_steps = int(final_time / dt)
        self.record_interval = max(1, int(record_interval))
        
        # Calculate exact number of snapshots to pre-allocate memory
        num_snapshots = (total_steps // self.record_interval) + 1
        if total_steps % self.record_interval != 0:
            num_snapshots += 1
            
        # Pre-allocate contiguous arrays
        self.time = np.zeros(num_snapshots)
        self.numerical = np.zeros((num_snapshots, grid_size))
        self.analytical = np.zeros((num_snapshots, grid_size))
        self.l1 = np.zeros(num_snapshots)
        self.l2 = np.zeros(num_snapshots)
        self.max_err = np.zeros(num_snapshots)
        
        self.idx = 0

    def record(self, t, num_state, anal_state):
        self.time[self.idx] = t
        self.numerical[self.idx, :] = num_state
        self.analytical[self.idx, :] = anal_state
        
        # Compute errors locally
        diff = num_state - anal_state
        self.l1[self.idx] = np.mean(np.abs(diff))
        self.l2[self.idx] = np.sqrt(np.mean(diff**2))
        self.max_err[self.idx] = np.max(np.abs(diff))
        
        self.idx += 1
        
    def get_history_dataframe(self):
        # Slice up to self.idx in case of early termination
        return pd.DataFrame({
            "time": self.time[:self.idx],
            "l2_error": self.l2[:self.idx],
            "l1_error": self.l1[:self.idx],
            "max_error": self.max_err[:self.idx]
        })