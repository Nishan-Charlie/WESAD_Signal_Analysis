import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def analyze_subjects_dist():
    WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'
    all_subjects = [f'S{i}' for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
    
    label_map = {1: 0, 3: 1, 2: 2} # baseline, amusement, stress
    class_names = {0: 'No Stress', 1: 'Low Stress', 2: 'High Stress'}
    
    subject_stats = []
    
    print("Extracting class distributions for all subjects...")
    for sid in all_subjects:
        file_path = os.path.join(WESAD_PATH, sid, f"{sid}.pkl")
        if not os.path.exists(file_path):
            print(f"Skipping {sid} (not found)")
            continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        labels = data['label']
        
        unique, counts = np.unique(labels, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        
        # Aggregate based on label_map
        mapped_counts = {0: 0, 1: 0, 2: 0}
        for wesad_label, target_label in label_map.items():
            mapped_counts[target_label] = counts_dict.get(wesad_label, 0)
        
        total_mapped = sum(mapped_counts.values())
        
        subject_stats.append({
            'Subject': sid,
            'No Stress': mapped_counts[0],
            'Low Stress': mapped_counts[1],
            'High Stress': mapped_counts[2],
            'Total': total_mapped
        })
        print(f"Completed {sid}")

    df = pd.DataFrame(subject_stats)
    df.to_csv('subject_class_distribution.csv', index=False)
    print("\nNumerical data saved to subject_class_distribution.csv")
    
    # Plotting
    subjects = df['Subject']
    # Use percentages for better comparison since total counts might differ
    p_no = df['No Stress'] / df['Total'] * 100
    p_low = df['Low Stress'] / df['Total'] * 100
    p_high = df['High Stress'] / df['Total'] * 100
    
    plt.figure(figsize=(14, 8))
    plt.bar(subjects, p_no, label='No Stress (Baseline)', color='#4A90E2')
    plt.bar(subjects, p_low, bottom=p_no, label='Low Stress (Amusement)', color='#50E3C2')
    plt.bar(subjects, p_high, bottom=p_no+p_low, label='High Stress (Stress)', color='#D0021B')
    
    plt.title('Percentage Class Distribution per Subject (WESAD)', fontsize=16)
    plt.ylabel('Percentage of Samples (%)', fontsize=12)
    plt.xlabel('Subject ID', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = 'subject_distribution_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    analyze_subjects_dist()
