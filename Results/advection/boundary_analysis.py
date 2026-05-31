import numpy as np
import matplotlib.pyplot as plt

frame = 250

numerical = np.load("Results/advection/numerical_states.npy")
analytic = np.load("Results/advection/analytic_states.npy")\
    
print(numerical.shape)
print(analytic.shape)

for frame in [245, 248, 250, 252, 255]:
    print(
        frame,
        np.argmax(numerical[frame]),
        np.argmax(analytic[frame])
    )

'''print("LEFT EDGE")
print("Numerical:", numerical[frame][:10])
print("Analytic :", analytic[frame][:10])

print("\nRIGHT EDGE")
print("Numerical:", numerical[frame][-10:])
print("Analytic :", analytic[frame][-10:])'''

    
plt.plot(numerical[frame], label="Numerical")
plt.plot(analytic[frame], label="Analytical")
plt.legend()
plt.grid()
plt.show()