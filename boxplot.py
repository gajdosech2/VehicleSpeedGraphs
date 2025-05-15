import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# Dummy data: Replace these with your actual speed errors (one array per system)
np.random.seed(42)  # For reproducibility
system_1 = np.random.normal(loc=0.8, scale=0.15, size=100)
system_2 = np.random.normal(loc=1.0, scale=0.2, size=100)
system_3 = np.random.normal(loc=0.9, scale=0.1, size=100)
system_4 = np.random.normal(loc=1.1, scale=0.25, size=100)
system_5 = np.random.normal(loc=0.85, scale=0.15, size=100)
system_6 = np.random.normal(loc=0.95, scale=0.12, size=100)

system_1 = np.loadtxt('SochorCVIU_Edgelets_ManualScale_errors_abs.txt', dtype=float).tolist()
system_2 = np.loadtxt('Transform3D_960_540_VP2VP3_errors_abs.txt', dtype=float).tolist()
system_3 = np.loadtxt('yolov6_nano_b32_640_352_fp32_titan_errors_abs.txt', dtype=float).tolist()
system_4 = np.loadtxt('yolov6_nano_b1_640_352_int8_jupiter_errors_abs.txt', dtype=float).tolist()
system_5 = np.loadtxt('yolov6_small_b32_640_352_fp32_titan_errors_abs.txt', dtype=float).tolist()
system_6 = np.loadtxt('yolov6_small_b1_640_352_int8_jupiter_errors_abs.txt', dtype=float).tolist()

# Combine all systems into a list for boxplotting
data = [system_1, system_2, system_3, system_4, system_5, system_6]
labels = ['Sochor\nManual', 'Transform3D\n960 x 540', 'Nano\n640 x 360\nFP32', 'Nano\n640 x 360\nINT8', 'Small\n640 x 360\nFP32', 'Small\n640 x 360\nINT8']

# Plot
plt.figure(figsize=(6, 5))
box = plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True, showfliers=False,
            boxprops=dict(facecolor='lightgray', color='lightgray'),
            medianprops=dict(color='gray', linewidth=2.5),
            meanprops=dict(marker='o', markerfacecolor='gray', markeredgecolor='gray', markersize=5),
            whiskerprops=dict(color='gray'),
            capprops=dict(color='gray'),
            flierprops=dict(marker='x', markerfacecolor='gray', markersize=5))


colors = ['lightblue', 'lightgreen', 'lightcoral', 'wheat', 'plum', 'red']
colormap = cm.get_cmap('YlOrRd') 
colors = ['lightblue', 'lightblue', colormap(3 / 3), colormap(1 / 3), colormap(3 / 3), colormap(1 / 3)]

#handles = []
#for patch, color, label in zip(box['boxes'], colors, labels):
#    patch.set_facecolor(color)
#    handles.append(mpatches.Patch(color=color, label=label))

#plt.xticks([])

#plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.2)) 
ax.grid(which='major', axis='y', linestyle='--', alpha=0.7)

#plt.xticks(rotation=-60)

plt.ylabel('Speed Error (km/h)')
#plt.title('Boxplot of Speed Estimation Errors')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'boxplot.pdf', bbox_inches='tight')
plt.show()
