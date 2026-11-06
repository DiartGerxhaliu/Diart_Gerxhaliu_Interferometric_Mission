import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from reference.esa_reference import udp


print("Matplotlib backend:", matplotlib.get_backend())

solution = np.load("results/best_solution.npz")

best_x = solution["x"]
best_fitness = float(solution["fitness"])

print("Saved fitness:", best_fitness)

reference_fitness = float(udp.fitness(best_x.tolist())[0])
print("Reference fitness:", reference_fitness)

print("\nGenerating formation plot...")

udp.plot(best_x)

print("Figures created:", plt.get_fignums())

for number in plt.get_fignums():
    figure = plt.figure(number)
    figure.show()

plt.show(block=True)

print("Plot window closed.")