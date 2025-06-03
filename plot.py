import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Without fine-tuning results (in percentage)
without_ft = {
    'Accuracy': 76.06,
    'Precision': 99,
    'Recall': 71.01,
    'F1-Score': 83.25
}

# CSV files with different percentages of data used for fine-tuning
csv_files = [
    '/home/tirthesh/chestxray/Code/result/metrics_1_100.csv', 
    '/home/tirthesh/chestxray/Code/result/metrics_15_100.csv', 
    '/home/tirthesh/chestxray/Code/result/metrics_01_100.csv', 
    '/home/tirthesh/chestxray/Code/result/metrics_5_100.csv'
]

# Lists to store accuracy, precision, recall, and f1-score data from all datasets
all_accuracies = []
all_precisions = []
all_recalls = []
all_f1_scores = []

# Load each CSV file and extract the metrics (multiply values by 100)
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    all_accuracies.append(df['Accuracy'].values * 100)
    all_precisions.append(df['Precision'].values * 100)
    all_recalls.append(df['Recall'].values * 100)
    all_f1_scores.append(df['F1-Score'].values * 100)

# Labels for the box plot (these will correspond to your datasets)
dataset_labels = ['1%', '0.5%', '0.1%', '0.075%']

# Function to plot four subplots for each metric
def plot_all_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Create a 2x2 grid of subplots
    fig.suptitle('Model Metrics vs. Percent of Data Used for Fine-Tuning', fontsize=16)
    
    # Plot positions
    positions = np.array(range(len(dataset_labels)))*2.0

    # Define the metrics for easier iteration
    metrics = {
        'Accuracy': {'data': all_accuracies, 'color': 'b', 'ax': axes[0, 0], 'without_ft': without_ft['Accuracy']},
        'Precision': {'data': all_precisions, 'color': 'g', 'ax': axes[0, 1], 'without_ft': without_ft['Precision']},
        'Recall': {'data': all_recalls, 'color': 'r', 'ax': axes[1, 0], 'without_ft': without_ft['Recall']},
        'F1-Score': {'data': all_f1_scores, 'color': 'y', 'ax': axes[1, 1], 'without_ft': without_ft['F1-Score']}
    }

    # Plot each metric
    for metric_name, metric_info in metrics.items():
        ax = metric_info['ax']
        data = metric_info['data']
        color = metric_info['color']
        without_ft_value = metric_info['without_ft']
        
        # Box plot
        ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, 
                   boxprops=dict(facecolor=color, color=color), medianprops=dict(color='black'))

        # Dashed line for "Without Fine-Tuning"
        ax.axhline(y=without_ft_value, color=color, linestyle='--', label=f'Without Fine-Tuning ({metric_name})')

        # Plot medians
        medians = [np.median(metric) for metric in data]
        ax.plot(positions, medians, f'{color}-', label=f'{metric_name} (median)')
        
        # Customize the subplot
        ax.set_title(metric_name)
        ax.set_xticks(positions)
        ax.set_xticklabels(dataset_labels)
        ax.set_ylabel('Metric (%)')
        ax.grid(True)
        ax.legend(loc='lower right')

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to fit the suptitle
    plt.savefig(os.path.join('result', 'metrics_subplots.png'))
    plt.show()

# Call the function to plot all metrics
plot_all_metrics()
