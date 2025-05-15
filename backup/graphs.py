import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle

# The provided configuration results dictionary
CONFIG_RESULTS = {
    "nano, 1int8, titanV, 640x360": (355, 0.71, 0.55, 91.63, 83.95),
    "nano, 2fp16, titanV, 640x360": (267, 0.80, 0.64, 92.41, 92.58),
    "nano, 3fp32, titanV, 640x360": (221, 0.80, 0.64, 92.42, 92.57),

    "small, 1int8, titanV, 640x360": (302, 0.80, 0.62, 96.24, 85.22),
    "small, 2fp16, titanV, 640x360": (205, 0.77, 0.58, 93.41, 92.48),
    "small, 3fp32, titanV, 640x360": (142, 0.77, 0.59, 93.40, 92.48),

    "medium, 2fp16, titanV, 640x360": (174, 0.86, 0.69, 93.15, 90.96),
    "medium, 3fp32, titanV, 640x360": (120, 0.86, 0.69, 93.13, 90.95),

    "large, 2fp16, titanV, 640x360": (159, 0.84, 0.66, 92.72, 90.94),
    "large, 3fp32, titanV, 640x360": (95, 0.84, 0.66, 92.75, 90.94),
}

def plot_results(fixed, shapes, colors, hw):
    # Define markers and colors
    markers = cycle(['p', 's', 'o', '^'])  # Cycle through markers
    colormap = cm.get_cmap('YlOrRd')  # Use a colormap for colors

    # Extract keys for unique identifiers
    all_keys = [key.split(', ') for key in CONFIG_RESULTS.keys()]
    fixed_values = sorted(set(key[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[fixed]] for key in all_keys))
    shape_values = sorted(set(key[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[shapes]] for key in all_keys))
    color_values = sorted(set(key[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[colors]] for key in all_keys))

    print(len(color_values))

    for fixed_val in fixed_values:
        plt.figure(figsize=(11, 5))
        current_markers = {shape: next(markers) for shape in shape_values}
        current_colors = {color: colormap((i+1) / len(color_values)) for i, color in enumerate(color_values)}

        for key, (fps, mean_error, median_error, precision, recall) in CONFIG_RESULTS.items():
            key_parts = key.split(', ')

            if (
                key_parts[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[fixed]] == fixed_val
                and key_parts[2] == hw  # Filter by hardware/gpu
            ):
                shape = key_parts[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[shapes]]
                color = key_parts[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[colors]]

                plt.scatter(fps, mean_error, label=f"{shape:<6} {color:<6}", 
                            color=current_colors[color], marker='o', s=400) #current_markers[shape]

        #plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        #plt.title(f"Graph for fixed {fixed}: {fixed_val} (HW: {hw})", fontsize=16)
        plt.xlabel(r"$\rightarrow$ FPS $\rightarrow$", fontsize=26)
        plt.ylabel("$\leftarrow$ Speed Error $\leftarrow$", fontsize=26)
        plt.legend(title=f"Marker Model Size dType", fontsize=14, title_fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5), prop={'family': 'monospace', 'size': 17})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=19)
        plt.show()
        #plt.savefig(f'graph{fixed_val}.pdf', bbox_inches='tight')
        #break

# Example usage
plot_results("input_size", "model_size", "dtype", "titanV")
#plot_results("dtype", "model_size", "input_size", "titanV")



# import matplotlib.pyplot as plt

# # Data extracted and organized into a dictionary
# model_data = {
#     "Nano": {"mAP": 45.0, "mAR": 67.4, "mean_error": 0.82, "median_error": 0.66},
#     "Nano distill": {"mAP": 46.0, "mAR": 67.0, "mean_error": 0.87, "median_error": 0.70},
#     "Small": {"mAP": 51.0, "mAR": 72.6, "mean_error": 0.81, "median_error": 0.63},
#     "Small distill": {"mAP": 50.5, "mAR": 68.9, "mean_error": 0.86, "median_error": 0.67},
#     "Medium": {"mAP": 51.0, "mAR": 71.4, "mean_error": 0.83, "median_error": 0.66},
#     "Large": {"mAP": 52.0, "mAR": 72.0, "mean_error": 0.82, "median_error": 0.64},
# }

# # Function to plot mAP vs mean speed error
# def plot_map_vs_mean_error(data):
#     # Define marker styles for different models
#     markers = {
#         "Nano": "o",  # square
#         "Nano distill": "o",  # square
#         "Small": "^",  # circle
#         "Small distill": "^",  # circle
#         "Medium": "s",  # diamond
#         "Large": "p",  # cross
#     }
#     colors = {
#         "Nano": "blue",
#         "Nano distill": "lightgreen",
#         "Small": "blue",
#         "Small distill": "lightgreen",
#         "Medium": "blue",
#         "Large": "blue",
#     }

#     plt.figure(figsize=(9, 6))

#     # Plot each model as a point
#     for model, values in data.items():
#         plt.scatter(
#             values["mAP"], values["mean_error"], 
#             label=model, marker=markers[model], color=colors[model], s=300
#         )
    
#     # Add labels and title
#     plt.xlabel("mAP (%) (Validation Set)", fontsize=16)
#     plt.ylabel("Mean Speed Error (km/h) (Test Set)", fontsize=16)
#     plt.title("mAP vs Mean Speed Error for Various Models", fontsize=16)
#     plt.legend(title="", fontsize=14, title_fontsize=14, loc='best')
#     plt.grid(True, linestyle="--", alpha=0.6)

#     # Show the plot
#     plt.savefig('map.png', bbox_inches='tight')
#     plt.show()

# # Call the function to plot the graph
# #plot_map_vs_mean_error(model_data)
