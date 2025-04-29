import numpy as np
import matplotlib.pyplot as plt

# Dummy data: Replace these with your actual speed errors (one array per system)
np.random.seed(42)  # For reproducibility
system_1 = np.random.normal(loc=0.8, scale=0.15, size=100)
system_2 = np.random.normal(loc=1.0, scale=0.2, size=100)
system_3 = np.random.normal(loc=0.9, scale=0.1, size=100)
system_4 = np.random.normal(loc=1.1, scale=0.25, size=100)
system_5 = np.random.normal(loc=0.85, scale=0.15, size=100)
system_6 = np.random.normal(loc=0.95, scale=0.12, size=100)

# Combine all systems into a list for boxplotting
data = [system_1, system_2, system_3, system_4, system_5, system_6]
labels = ['System A', 'System B', 'System C', 'System D', 'System E', 'System F']

# Plot
plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'),
            meanprops=dict(marker='o', markerfacecolor='black', markersize=5),
            whiskerprops=dict(color='gray'),
            capprops=dict(color='gray'),
            flierprops=dict(marker='x', markerfacecolor='gray', markersize=5))

plt.ylabel('Speed Error (km/h)')
plt.title('Comparison of Speed Estimation Errors Across Systems')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
