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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

def get_loso_split(fold_idx, all_subjects):
    """
    LOSO Split for 15 subjects.
    Test: 1 subject (index fold_idx)
    Val: 1 subject (index fold_idx + 1 mod 15)
    Train: 13 subjects (others)
    """
    n = len(all_subjects)
    test_subject = all_subjects[fold_idx]
    val_subset = [all_subjects[(fold_idx + i + 1) % n] for i in range(1)] # Using 1 subject for validation (~7%)
    
    val_subjects = val_subset
    train_subjects = [s for s in all_subjects if s != test_subject and s not in val_subjects]
    
    return train_subjects, val_subjects, [test_subject]

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
        "f1": f1,
        "report": classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    }

def train_fold(fold_idx, train_subjects, val_subjects, test_subjects, args):
    WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'
    OUTPUT_FOLDER = os.path.join("output", "loso", f"fold_{fold_idx}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*50)
    print(f"LOSO FOLD {fold_idx}/14 | TEST SUBJECT: {test_subjects[0]}")
    print(f"Train on {len(train_subjects)} subjects, Validate on {len(val_subjects)} subject")
    print("="*50)
    
    # Data Loading (Reuse WESADDataset)
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
    
    best_val_f1 = 0
    history = {"train_loss": [], "val_f1": [], "val_acc": []}
    
    for epoch in range(args['epochs']):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        for ecg, eda, emg, resp, label in train_loader:
            ecg, eda, emg, resp, label = ecg.to(device), eda.to(device), emg.to(device), resp.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(ecg, eda, emg, resp)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        val_m = evaluate(model, val_loader, device)
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_f1"].append(val_m["f1"])
        history["val_acc"].append(val_m["accuracy"])
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{args['epochs']} | Loss: {history['train_loss'][-1]:.4f} | Val F1: {val_m['f1']:.4f} | Acc: {val_m['accuracy']:.4f} | {elapsed:.1f}s")
        
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER, "best_model.pth"))
            
    # Final Testing on the Left-Out Subject
    model.load_state_dict(torch.load(os.path.join(OUTPUT_FOLDER, "best_model.pth")))
    test_m = evaluate(model, test_loader, device)
    
    print(f"\n>>> FOLD {fold_idx} TEST RESULTS ({test_subjects[0]}): Accuracy: {test_m['accuracy']:.4f}, F1-Macro: {test_m['f1']:.4f}")
    
    results = {
        "fold": fold_idx,
        "test_subject": test_subjects[0],
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "history": history,
        "test_metrics": test_m
    }
    
    with open(os.path.join(OUTPUT_FOLDER, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    return test_m

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--folds", type=int, default=15)
    parser.add_argument("--start_fold", type=int, default=0)
    args_in = parser.parse_args()
    
    all_subjects = [f'S{i}' for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
    
    args = {
        "epochs": args_in.epochs,
        "batch_size": args_in.batch_size,
        "lr": args_in.lr
    }
    
    all_fold_metrics = []
    
    for f in range(args_in.start_fold, args_in.folds):
        tr, vl, ts = get_loso_split(f, all_subjects)
        m = train_fold(f, tr, vl, ts, args)
        all_fold_metrics.append(m)
        
    # Aggregate Summary
    if len(all_fold_metrics) > 0:
        avg_acc = np.mean([m['accuracy'] for m in all_fold_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_fold_metrics])
        avg_pre = np.mean([m['precision'] for m in all_fold_metrics])
        avg_rec = np.mean([m['recall'] for m in all_fold_metrics])
        
        summary = {
            "avg_accuracy": avg_acc,
            "avg_f1_macro": avg_f1,
            "avg_precision_macro": avg_pre,
            "avg_recall_macro": avg_rec,
            "folds_completed": len(all_fold_metrics)
        }
        
        with open("output/loso/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        print("\n" + "="*50)
        print("OVERALL LOSO RESULTS")
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Average F1-Macro: {avg_f1:.4f}")
        print(f"Summary saved to output/loso/summary.json")
        print("="*50)

if __name__ == "__main__":
    main()
