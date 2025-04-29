import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Dummy data: Replace these with your actual speed errors (one array per system)
np.random.seed(42)  # For reproducibility
system_1 = np.random.normal(loc=0.8, scale=0.15, size=100)
system_2 = np.random.normal(loc=1.0, scale=0.2, size=100)
system_3 = np.random.normal(loc=0.9, scale=0.1, size=100)
system_4 = np.random.normal(loc=1.1, scale=0.25, size=100)
system_5 = np.random.normal(loc=0.85, scale=0.15, size=100)
system_6 = np.random.normal(loc=0.95, scale=0.12, size=100)

system_1 = np.loadtxt('yolov6_nano_23_b32_fp32_titan_errors_abs.txt', dtype=float).tolist()
system_2 = np.loadtxt('yolov6_distill_nano_23_b32_fp32_titan_errors_abs.txt', dtype=float).tolist()
system_3 = np.loadtxt('yolov6_small_23_b32_fp32_titan_errors_abs.txt', dtype=float).tolist()
system_4 = np.loadtxt('yolov6_small_distill_23_fp16_fp32_titan_errors_abs.txt', dtype=float).tolist()
system_5 = np.loadtxt('yolov6_medium_23_b32_fp32_titan_errors_abs.txt', dtype=float).tolist()
system_6 = np.loadtxt('yolov6_large_23_b32_fp32_titan_errors_abs.txt', dtype=float).tolist()

# Combine all systems into a list for boxplotting
data = [system_1, system_2, system_3, system_4, system_5, system_6]
labels = ['Nano', 'Nano Distill', 'Small', 'Small Distill', 'Medium', 'Large']

# Plot
plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True, showfliers=False,
            boxprops=dict(facecolor='lightblue', color='lightblue'),
            medianprops=dict(color='red'),
            meanprops=dict(marker='o', markerfacecolor='green', markersize=5),
            whiskerprops=dict(color='gray'),
            capprops=dict(color='gray'),
            flierprops=dict(marker='x', markerfacecolor='gray', markersize=5))

ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.1)) 
ax.grid(which='major', axis='y', linestyle='--', alpha=0.7)

plt.ylabel('Speed Error (km/h)')
plt.title('Boxplot of Speed Estimation Errors')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
