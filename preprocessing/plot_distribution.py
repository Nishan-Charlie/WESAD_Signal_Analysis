import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np
from wesad_dataset import WESADDataset
import torch

def plot_distribution():
    WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'
    
    # Same splits as train_wesad.py
    all_subjects = [f'S{i}' for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
    train_subjects = all_subjects[:12]
    val_subjects = all_subjects[12:]
    
    print("Loading Train Dataset...")
    train_dataset = WESADDataset(WESAD_PATH, train_subjects, window_size=700, step_size=350)
    
    print("Loading Validation Dataset...")
    val_dataset = WESADDataset(WESAD_PATH, val_subjects, window_size=700, step_size=700)
    
    def get_counts(dataset):
        labels = dataset.labels.numpy()
        unique, counts = np.unique(labels, return_counts=True)
        # Ensure all classes 0, 1, 2 are present even if 0
        dist = {0: 0, 1: 0, 2: 0}
        for u, c in zip(unique, counts):
            dist[u] = c
        return dist

    train_dist = get_counts(train_dataset)
    val_dist = get_counts(val_dataset)
    
    classes = ['No Stress', 'Low Stress', 'High Stress']
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, [train_dist[i] for i in range(3)], width, label='Train', color='#4A90E2')
    rects2 = ax.bar(x + width/2, [val_dist[i] for i in range(3)], width, label='Validation', color='#50E3C2')
    
    ax.set_ylabel('Number of Samples')
    ax.set_title('WESAD Data Distribution (Class Counts)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    
    # Add counts on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plot_path = os.path.join(os.getcwd(), 'data_distribution.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_distribution()
