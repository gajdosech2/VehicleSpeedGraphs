import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle

# The provided configuration results dictionary
CONFIG_RESULTS = {
    "nano, int8, titanV, 480x270": (464, 0.89, 0.69, 88.05, 91.51),
    "nano, int8, titanV, 640x360": (355, 0.82, 0.65, 89.62, 91.27),
    "nano, int8, titanV, 960x540": (246, 0.87, 0.69, 90.92, 90.13),

    "nano, fp16, titanV, 480x270": (357, 0.87, 0.70, 88.08, 91.23),
    "nano, fp16, titanV, 640x360": (267, 0.80, 0.64, 89.69, 92.25),
    "nano, fp16, titanV, 960x540": (179, 0.85, 0.66, 90.60, 92.74),

    "nano, fp32, titanV, 480x270": (295, 0.87, 0.69, 88.08, 91.23),
    "nano, fp32, titanV, 640x360": (221, 0.80, 0.64, 89.69, 92.25),
    "nano, fp32, titanV, 960x540": (132, 0.82, 0.66, 91.15, 90.10),

    "nano, int8, orin, 480x270": (95, 0.89, 0.69, 88.05, 91.51),
    "nano, int8, orin, 640x360": (68, 0.82, 0.65, 89.62, 91.27),
    "nano, int8, orin, 960x540": (57, 0.87, 0.69, 90.92, 90.13),

    "nano, fp16, orin, 480x270": (76, 0.87, 0.70, 88.08, 91.23),
    "nano, fp16, orin, 640x360": (54, 0.80, 0.64, 89.69, 92.25),
    "nano, fp16, orin, 960x540": (38, 0.85, 0.66, 90.60, 92.74),

    "nano, fp32, orin, 480x270": (65, 0.87, 0.69, 88.08, 91.23),
    "nano, fp32, orin, 640x360": (43, 0.80, 0.64, 89.69, 92.25),
    "nano, fp32, orin, 960x540": (31, 0.82, 0.66, 91.15, 90.10),


    "small, int8, titanV, 480x270": (392, 0.88, 0.70, 88.08, 91.23),
    "small, int8, titanV, 640x360": (302, 0.78, 0.60, 91.02, 91.27),
    "small, int8, titanV, 960x540": (198, 0.83, 0.62, 90.60, 90.07),

    "small, fp16, titanV, 480x270": (271, 0.87, 0.68, 91.08, 91.85),
    "small, fp16, titanV, 640x360": (205, 0.76, 0.58, 91.02, 92.16),
    "small, fp16, titanV, 960x540": (139, 0.81, 0.62, 92.09, 91.86),

    "small, fp32, titanV, 480x270": (223, 0.87, 0.68, 91.08, 92.85),
    "small, fp32, titanV, 640x360": (142, 0.81, 0.58, 91.02, 92.16),
    "small, fp32, titanV, 960x540": (98, 0.81, 0.63, 92.11, 91.14),

    "small, int8, orin, 480x270": (126, 0.88, 0.70, 88.08, 91.23),
    "small, int8, orin, 640x360": (90, 0.78, 0.60, 91.02, 91.27),
    "small, int8, orin, 960x540": (95, .83, 0.62, 90.60, 90.07),

    "small, fp16, orin, 480x270": (58, 0.87, 0.68, 91.08, 91.85),
    "small, fp16, orin, 640x360": (39, 0.76, 0.58, 91.02, 92.16),
    "small, fp16, orin, 960x540": (24, 0.81, 0.62, 92.09, 91.86),

    "small, fp32, orin, 480x270": (45, 0.87, 0.68, 91.08, 92.85),
    "small, fp32, orin, 640x360": (30, 0.81, 0.58, 91.02, 92.16),
    "small, fp32, orin, 960x540": (18, 0.81, 0.63, 92.11, 91.14),


    "medium, fp16, titanV, 480x270": (330, 0.84, 0.67, 91.55, 87.44),
    "medium, fp16, titanV, 640x360": (174, 0.86, 0.69, 91.32, 90.68),
    "medium, fp16, titanV, 960x540": (108, 0.82, 0.63, 91.23, 91.01),

    "medium, fp32, titanV, 480x270": (162, 0.84, 0.67, 91.55, 92.44),
    "medium, fp32, titanV, 640x360": (120, 0.86, 0.69, 91.32, 90.68),
    "medium, fp32, titanV, 960x540": (67, 0.83, 0.66, 91.10, 91.00),

    "medium, fp16, orin, 480x270": (39, 0.84, 0.67, 91.55, 87.44),
    "medium, fp16, orin, 640x360": (24, 0.86, 0.69, 91.32, 90.68),
    "medium, fp16, orin, 960x540": (14, 0.82, 0.63, 91.23, 91.01),

    "medium, fp32, orin, 480x270": (31, 0.84, 0.67, 91.55, 92.44),
    "medium, fp32, orin, 640x360": (19, 0.86, 0.69, 91.32, 90.68),
    "medium, fp32, orin, 960x540": (9, 0.82, 0.63, 91.23, 91.01),


    "large, fp16, titanV, 480x270": (354, 0.87, 0.70, 90.85, 91.12),
    "large, fp16, titanV, 640x360": (159, 0.84, 0.66, 90.34, 90.67),
    "large, fp16, titanV, 960x540": (84, 0.81, 0.62, 91.11, 89.61),

    "large, fp32, titanV, 480x270": (125, 0.87, 0.70, 90.85, 91.12),
    "large, fp32, titanV, 640x360": (95, 0.84, 0.66, 90.34, 90.67),
    "large, fp32, titanV, 960x540": (50, 0.82, 0.64, 91.21, 90.85),

    "large, fp16, orin, 480x270": (29, 0.87, 0.70, 90.85, 91.12),
    "large, fp16, orin, 640x360": (18, 0.84, 0.66, 90.34, 90.67),
    "large, fp16, orin, 960x540": (10, 0.81, 0.62, 91.11, 89.61),

    "large, fp32, orin, 480x270": (21, 0.87, 0.70, 90.85, 91.12),
    "large, fp32, orin, 640x360": (14, 0.84, 0.66, 90.34, 90.67),
    "large, fp32, orin, 960x540": (7, 0.82, 0.64, 91.21, 90.85),
}

def plot_results(fixed, shapes, colors, hw):
    # Define markers and colors
    markers = cycle(['o', 's', '^', 'P'])  # Cycle through markers
    colormap = cm.get_cmap('tab10')  # Use a colormap for colors

    # Extract keys for unique identifiers
    all_keys = [key.split(', ') for key in CONFIG_RESULTS.keys()]
    fixed_values = sorted(set(key[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[fixed]] for key in all_keys))
    shape_values = sorted(set(key[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[shapes]] for key in all_keys))
    color_values = sorted(set(key[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[colors]] for key in all_keys))

    for fixed_val in fixed_values:
        plt.figure(figsize=(10, 6))
        current_markers = {shape: next(markers) for shape in shape_values}
        current_colors = {color: colormap(i / len(color_values)) for i, color in enumerate(color_values)}

        for key, (fps, mean_error, median_error, precision, recall) in CONFIG_RESULTS.items():
            key_parts = key.split(', ')

            if (
                key_parts[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[fixed]] == fixed_val
                and key_parts[2] == hw  # Filter by hardware/gpu
            ):
                shape = key_parts[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[shapes]]
                color = key_parts[{'dtype': 1, 'model_size': 0, 'input_size': 3, 'gpu': 2}[colors]]

                plt.scatter(fps, median_error, label=f"{shape}, {color}", 
                            color=current_colors[color], marker=current_markers[shape], s=100)

        plt.title(f"Graph for fixed {fixed}: {fixed_val} (HW: {hw})", fontsize=16)
        plt.xlabel("FPS", fontsize=14)
        plt.ylabel("Speed Error", fontsize=14)
        plt.legend(title=f"Shapes ({shapes}) and Colors ({colors})", fontsize=10, title_fontsize=12, loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

# Example usage
plot_results("dtype", "model_size", "input_size", "orin")



# Function to plot the FPS based on a given hardware and model
def plot_fps_for_hardware(hardware_name):
    # Extract configurations that match the given hardware name
    configurations = [key for key in CONFIG_RESULTS.keys() if hardware_name in key]
    
    # Prepare FPS data and corresponding configuration labels
    fps_values = [CONFIG_RESULTS[config][0] for config in configurations]  # Extract FPS values
    labels = [config.replace(f"{hardware_name}, ", "") for config in configurations]  # Format labels

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, fps_values, color='blue')

    # Labeling the chart
    ax.set_title(f"FPS for {hardware_name}")
    ax.set_xlabel("Configuration (Model, Data Type, Resolution)")
    ax.set_ylabel("FPS")
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Add FPS values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 2, round(yval, 2), ha="center", va="bottom")

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
plot_fps_for_hardware("titanV")



import matplotlib.pyplot as plt
import numpy as np

# Extracted dictionary from the table
DISTILLED_CONFIGS = {
    "Nano, 480x270": (0.87, 0.69, 88.08, 91.23),
    "Nano_distill, 480x270": (0.98, 0.74, 81.71, 87.39),
    "Nano, 640x360": (0.80, 0.64, 89.69, 92.25),
    "Nano_distill, 640x360": (0.90, 0.81, 86.00, 89.15),
    "Nano, 960x540": (0.82, 0.66, 91.15, 90.10),
    "Nano_distill, 960x540": (0.87, 0.70, 84.72, 86.53),
    "Small, 480x270": (0.87, 0.68, 91.08, 92.85),
    "Small_distill, 480x270": (0.91, 0.73, 85.45, 91.00),
    "Small, 640x360": (0.81, 0.58, 91.02, 92.16),
    "Small_distill, 640x360": (0.84, 0.65, 85.05, 89.58),
    "Small, 960x540": (0.81, 0.63, 92.11, 91.14),
    "Small_distill, 960x540": (0.86, 0.67, 84.73, 90.07),
}

# Plotting function
def plot_accuracy_comparison(input_resolution):
    models = []
    metrics = {
        "Mean Error": [],
        "Median Error": [],
 #       "Precision": [],
 #       "Recall": []
    }
    
    # Extract relevant data for the input resolution
    for key, values in DISTILLED_CONFIGS.items():
        model, resolution = key.split(", ")
        if resolution == input_resolution:
            models.append(model)
            metrics["Mean Error"].append(values[0])
            metrics["Median Error"].append(values[1])
            #metrics["Precision"].append(values[2])
            #metrics["Recall"].append(values[3])
    
    if not models:
        print(f"No data found for input resolution: {input_resolution}")
        return
    
    # Bar positions
    x = np.arange(len(models) // 2)  # Groups (Nano, Nano_distill, etc.)
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        # Separate normal and distilled values
        original_values = metric_values[::2]  # Even indices
        distilled_values = metric_values[1::2]  # Odd indices
        
        # Bar positions for normal and distilled
        bar_pos_original = x + (i - 1.5) * width / 4
        bar_pos_distilled = bar_pos_original + width / 2
        
        ax.bar(
            bar_pos_original,
            original_values,
            width / 4,
            label=f"{metric_name} (Original)",
            alpha=0.75
        )
        ax.bar(
            bar_pos_distilled,
            distilled_values,
            width / 4,
            label=f"{metric_name} (Distilled)",
            alpha=0.75
        )

    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels([model.split("_")[0] for model in models[::2]], rotation=45)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Accuracy Comparison for Input Resolution {input_resolution}")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    
    # Show the plot
    plt.show()

# Example usage
plot_accuracy_comparison("960x540")
