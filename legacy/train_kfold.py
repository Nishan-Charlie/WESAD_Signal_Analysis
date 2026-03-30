import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from wesad_dataset import WESADDataset
from classical_baseline import ClassicalBaseline
import time
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def get_subject_split(fold_idx, all_subjects):
    """
    5-Fold Subject-Level Split for 15 subjects.
    Each fold: 3 Test, 1 Val, 11 Train.
    """
    n = len(all_subjects)
    test_size = 3
    test_start = fold_idx * test_size
    test_subjects = all_subjects[test_start : test_start + test_size]
    
    # Validation is one subject before the test set
    val_idx = (test_start - 1) % n
    val_subjects = [all_subjects[val_idx]]
    
    train_subjects = [s for s in all_subjects if s not in test_subjects and s not in val_subjects]
    
    return train_subjects, val_subjects, test_subjects

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for ecg, eda, emg, resp, label in loader:
            ecg, eda, emg, resp, label = ecg.to(device), eda.to(device), emg.to(device), resp.to(device), label.to(device)
            outputs = model(ecg, eda, emg, resp)
            loss = criterion(outputs, label)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        "loss": total_loss / len(loader),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_fold(fold_idx, train_subjects, val_subjects, test_subjects, args):
    WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'
    OUTPUT_DIR = os.path.join("output", f"fold_{fold_idx}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Starting Fold {fold_idx} ---")
    print(f"Train: {train_subjects}\nVal: {val_subjects}\nTest: {test_subjects}")
    
    # Data Loading
    train_ds = WESADDataset(WESAD_PATH, train_subjects, window_size=700, step_size=350)
    val_ds = WESADDataset(WESAD_PATH, val_subjects, window_size=700, step_size=700)
    test_ds = WESADDataset(WESAD_PATH, test_subjects, window_size=700, step_size=700)
    
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args['batch_size'], shuffle=False)
    
    # Model
    model = ClassicalBaseline(latent_dim=8, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = 0
    
    for epoch in range(args['epochs']):
        model.train()
        epoch_loss = 0
        for ecg, eda, emg, resp, label in train_loader:
            ecg, eda, emg, resp, label = ecg.to(device), eda.to(device), emg.to(device), resp.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(ecg, eda, emg, resp)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        
        print(f"Epoch {epoch+1}/{args['epochs']} | Train Loss: {history['train_loss'][-1]:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            
    # Final Testing
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    test_metrics = evaluate(model, test_loader, device)
    
    results = {
        "fold": fold_idx,
        "subjects": {"train": train_subjects, "val": val_subjects, "test": test_subjects},
        "test_metrics": test_metrics,
        "history": history
    }
    
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    return test_metrics

def main():
    args = {
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 64,
        "folds": 5
    }
    
    all_subjects = [f'S{i}' for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
    fold_results = []
    
    for f in range(args['folds']):
        train_sub, val_sub, test_sub = get_subject_split(f, all_subjects)
        metrics = train_fold(f, train_sub, val_sub, test_sub, args)
        fold_results.append(metrics)
        
    # Aggregate results
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    
    print("\n" + "="*30)
    print("FINAL K-FOLD RESULTS")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")
    print("Detailed logs saved in /output directory.")
    print("="*30)

if __name__ == "__main__":
    main()
