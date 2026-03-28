import torch
import torch.nn as nn
import torch.optim as optim
from model import MultimodalQuantumFusion
import time

def train():
    # Hyperparameters
    BATCH_SIZE = 16 # Small batch for demo
    EPOCHS = 5
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 100
    
    print("Initializing Model...")
    model = MultimodalQuantumFusion(n_qubits=8, n_layers=3, n_classes=3)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model Initialized. Params: {sum(p.numel() for p in model.parameters())}")
    
    # Dummy Data Generation
    # Classes: 0: Baseline, 1: Stress, 2: Amusement
    print("Generating Dummy Data...")
    num_samples = 100
    
    # Inputs: (NumSamples, 1, Length)
    x_ecg = torch.randn(num_samples, 1, SEQ_LENGTH)
    x_eda = torch.randn(num_samples, 1, SEQ_LENGTH)
    x_emg = torch.randn(num_samples, 1, SEQ_LENGTH)
    x_resp = torch.randn(num_samples, 1, SEQ_LENGTH)
    
    y = torch.randint(0, 3, (num_samples,))
    
    # Dataset / Loader
    dataset = torch.utils.data.TensorDataset(x_ecg, x_eda, x_emg, x_resp, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (ecg, eda, emg, resp, target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward
            outputs = model(ecg, eda, emg, resp)
            
            # Loss
            loss = criterion(outputs, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {epoch_time:.2f}s")

    print("Training Complete.")

if __name__ == "__main__":
    train()
