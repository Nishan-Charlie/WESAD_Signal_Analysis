import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_loso_results():
    base_path = r'output/loso'
    plots_path = os.path.join(base_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)
    
    folds = sorted([d for d in os.listdir(base_path) if d.startswith('fold_')], key=lambda x: int(x.split('_')[1]))
    
    test_accs = []
    test_f1s = []
    subject_names = []
    
    print(f"Plotting results for {len(folds)} folds...")
    
    for fold in folds:
        json_path = os.path.join(base_path, fold, 'results.json')
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        history = data.get('history', {})
        subject = data.get('test_subject', fold)
        subject_names.append(subject)
        
        test_m = data.get('test_metrics', {})
        test_accs.append(test_m.get('accuracy', 0))
        test_f1s.append(test_m.get('f1', 0))
        
        # Plotting History (Loss, Acc, F1)
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Loss on Left Y-axis
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color=color)
        ax1.plot(epochs, history.get('train_loss', []), color=color, label='Train Loss', marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Accuracy/F1 on Right Y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Metrics', color='tab:blue')
        ax2.plot(epochs, history.get('val_acc', []), color='tab:blue', label='Val Acc', marker='s', alpha=0.7)
        ax2.plot(epochs, history.get('val_f1', []), color='tab:green', label='Val F1', marker='^', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        plt.title(f'Training Metrics - {fold} (Test Subject: {subject})')
        fig.tight_layout()
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_path, f'{fold}_metrics.png'))
        plt.close()
        
    # --- Summary Comparison Plot ---
    if subject_names:
        plt.figure(figsize=(15, 7))
        x = np.arange(len(subject_names))
        
        plt.bar(x - 0.2, test_accs, 0.4, label='Test Accuracy', color='#4A90E2', alpha=0.8)
        plt.bar(x + 0.2, test_f1s, 0.4, label='Test F1-Macro', color='#50E3C2', alpha=0.8)
        
        plt.xlabel('Test Subject')
        plt.ylabel('Score (0.0 - 1.0)')
        plt.title('Performance Across All 15 LOSO Folds')
        plt.xticks(x, subject_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.1)
        
        # Add labels on top of bars
        for i, acc in enumerate(test_accs):
            plt.text(i - 0.2, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
        for i, f1 in enumerate(test_f1s):
            plt.text(i + 0.2, f1 + 0.02, f'{f1:.2f}', ha='center', va='bottom', fontsize=8)
            
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, 'loso_summary_performance.png'))
        plt.close()
        
        print(f"All plots saved to: {plots_path}")

if __name__ == "__main__":
    plot_loso_results()
