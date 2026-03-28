import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from wesad_dataset import WESADDataset
from classical_baseline import ClassicalBaseline
import time
import argparse

def main(args):
    # Hyperparameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    
    # Path configuration
    WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'
    
    # 1. Subject Selection
    # Standard subjects in WESAD: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    all_subjects = [f'S{i}' for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
    
    if args.demo:
        # Use only 2 subjects for a quick walkthrough
        train_subjects = ['S2']
        val_subjects = ['S3']
        print(f"--- DEMO MODE ACTIVATED ---")
    else:
        # Simple split: first 12 subjects for train, last 3 for val
        train_subjects = all_subjects[:12]
        val_subjects = all_subjects[12:]
    
    print(f"Training on: {train_subjects}")
    print(f"Validation on: {val_subjects}")
    
    # 2. Data Loading
    train_dataset = WESADDataset(WESAD_PATH, train_subjects, window_size=700, step_size=350)
    val_dataset = WESADDataset(WESAD_PATH, val_subjects, window_size=700, step_size=700)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Total Train Samples: {len(train_dataset)}")
    print(f"Total Val Samples: {len(val_dataset)}")
    
    # 3. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ClassicalBaseline(latent_dim=8, num_classes=3).to(device)
    
    # Loss and Optimizer
    # nn.CrossEntropyLoss expects raw logits, which matches the updated ClassicalBaseline model.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    best_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for ecg, eda, emg, resp, label in train_loader:
            ecg, eda, emg, resp, label = ecg.to(device), eda.to(device), emg.to(device), resp.to(device), label.to(device)
            
            optimizer.zero_grad()
            outputs = model(ecg, eda, emg, resp)
            
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for ecg, eda, emg, resp, label in val_loader:
                ecg, eda, emg, resp, label = ecg.to(device), eda.to(device), emg.to(device), resp.to(device), label.to(device)
                outputs = model(ecg, eda, emg, resp)
                _, predicted = torch.max(outputs.data, 1)
                v_total += label.size(0)
                v_correct += (predicted == label).sum().item()
        
        val_acc = 100 * v_correct / v_total
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_classical_baseline.pth")
            print(f"--> Saved New Best Model (Acc: {best_acc:.2f}%)")

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run with small subset of subjects for speed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    main(args)
