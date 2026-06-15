# Agent Strategy Ledger

This ledger tracks the autonomous exploration of the 1D CNN surrogate model for linear advection.

## Success Conditions
1. Amplitude retention >= 90% (T=500).
2. Energy within +/- 5% of T=0.
3. Rightward propagation (Index of peak at T=500 > T=0).

## Hypothesis Tracker

### Iteration 0-6: 
**Status:** Completed. 

### Iteration 7: Hardmax / High-Temperature Extractor
**Status:** Completed. Increasing the inverse temperature from 5.0 to 100.0 worked wonders: `advec_peak` amplitude retention at T=500 jumped from 10.72% to 66.32%! And energy conservation is mathematically perfect (0.00% change). The model successfully sacrificed fractional phase interpolation for amplitude preservation.

### Iteration 8: Absolute Hardmax
**Hypothesis:** An inverse temperature of 100.0 was strong enough to push retention to 66%, but it still left a microscopic fraction of probability mass (e.g. $10^{-4}$) on adjacent weights. Unrolled 500 or 2000 times, even $10^{-4}$ leakage acts as a low-pass filter on an impulse shape. By cranking the inverse temperature to `1000.0`, the Softmax will output a mathematically absolute one-hot delta vector, totally eliminating the low-pass leakage and allowing the model to hit the >=90% amplitude target.
**Status:** In Progress.
