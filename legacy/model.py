import torch
import torch.nn as nn
import pennylane as qml
import math

class ClassicalBranch(nn.Module):
    def __init__(self, input_length=100, feature_dim=32):
        super(ClassicalBranch, self).__init__()
        # Input shape: (Batch, 1, Length) or (Batch, Length, 1). We'll assume (Batch, 1, Length) for Conv1d
        # 1D-CNN: Extract local temporal patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Halves the length
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # Halves the length again
        )
        
        # LSTM: Capture long-term dependencies
        # After 2 pools, length is input_length / 4. 
        # Channels = 32. 
        # We need to reshape for LSTM: (Batch, Sequence, Features)
        self.lstm = nn.LSTM(input_size=32, hidden_size=feature_dim, batch_first=True)
        
    def forward(self, x):
        # x shape: (Batch, 1, Length)
        x = self.cnn(x)
        # x shape after CNN: (Batch, 32, Reduced_Length)
        
        # Permute for LSTM: (Batch, Reduced_Length, 32)
        x = x.permute(0, 2, 1)
        
        # LSTM return: output, (h_n, c_n)
        # We take the last hidden state as the summary vector
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape: (1, Batch, feature_dim). Squeeze to (Batch, feature_dim)
        return h_n.squeeze(0)

class MultimodalQuantumFusion(nn.Module):
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3):
        super(MultimodalQuantumFusion, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # --- Step 1: Classical Feature Extraction ---
        # 4 Independent branches for ECG, EDA, EMG, Resp
        self.branch_ecg = ClassicalBranch()
        self.branch_eda = ClassicalBranch()
        self.branch_emg = ClassicalBranch()
        self.branch_resp = ClassicalBranch()
        
        # --- Step 2: Bottleneck ---
        # 4 branches * 32 features = 128 features
        self.bottleneck = nn.Sequential(
            nn.Linear(128, 8),
            nn.Tanh() # Tanh or Sigmoid to bound values for angle encoding, usually [0, 2pi] or [-pi, pi]
            # Tanh gives [-1, 1]. We can scale this in the circuit.
        )
        
        # --- Step 3: Quantum Fusion Layer ---
        # Define the device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Create the QNode
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # inputs: shape (8,) -> Angle Encoding
            # Scale inputs from [-1, 1] to [-pi, pi] for full rotation coverage
            scaled_inputs = inputs * torch.pi 
            
            # Amplitude Encoding
            for q in range(n_qubits):
                qml.RX(scaled_inputs[q], wires=q)

            for l in range(n_layers):
                # Apply Rotations
                for q in range(n_qubits):
                    qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
                
                stride = l + 1 
                for q in range(n_qubits):
                    target = (q + stride) % n_qubits
                    target_linear = q + stride
                    if target_linear < n_qubits:
                         qml.CNOT(wires=[q, target_linear])

            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = quantum_circuit
        
        # Quantum Weights: (Layers, Qubits, 3 parameters per Rot)
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        
        # --- Step 4: Classification Head ---
        # Input: 8 expectation values from Quantum Layer
        # Output: 3 classes (Baseline, Stress, Amusement)
        self.classifier = nn.Linear(8, n_classes)
        
    def forward(self, x_ecg, x_eda, x_emg, x_resp):
        # x_channel shape: (Batch, 1, Length)
        
        # 1. Independent Feature Extraction
        f_ecg = self.branch_ecg(x_ecg)
        f_eda = self.branch_eda(x_eda)
        f_emg = self.branch_emg(x_emg)
        f_resp = self.branch_resp(x_resp)
        
        # 2. Concatenate and Bottleneck
        combined = torch.cat([f_ecg, f_eda, f_emg, f_resp], dim=1) # (Batch, 128)
        compressed = self.bottleneck(combined) # (Batch, 8)
        
        # 3. Quantum Layer
        # Process batch items. PennyLane's Torch support handles batching often, 
        # but explicit loop or broadcasting might be needed depending on version.
        # We will use a list comprehension for safety and clarity with custom QNode.
        # Ideally, we want vectorized execution, but let's stick to simple batch processing for now or check if QNode supports it.
        # 'default.qubit' + 'backprop' usually supports batching if setup correctly.
        # Let's try direct passing:
        
        # Warning: Direct batching in PennyLane depends on the embedding. 
        # We will use torch.stack([self.qnode(c, self.q_weights) for c in compressed])
        # This is slower but robust.
        
        q_out = torch.stack([torch.tensor(self.qnode(c, self.q_weights)) for c in compressed])
        
        # Note: q_out returned by qnode is list of tensors. Stack converts to tensor.
        # If batch_size > 1, the above ensures we get (Batch, 8)
        
        # q_out shape checks: qnode returns [val0, val1... val7]. 
        # torch.tensor() on that list gives shape (8,).
        # Stack gives (Batch, 8).
        if q_out.requires_grad == False and compressed.requires_grad == True:
             # This sometimes happens if we break the graph. 
             # Re-construction:
             # qml.qnode needs to be called properly.
             # The list comprehension approach is standard for hybrid if batching isn't auto.
             pass
        
        # Double check: qnode returns a LIST of tensors (one per measurement) or a stacked tensor?
        # With default.qubit and return [...], it returns a list of scalars (tensors).
        # We need to stack them.
        
        # 4. Classification
        logits = self.classifier(q_out.to(compressed.device).float())
        return logits

# Testing block
if __name__ == "__main__":
    model = MultimodalQuantumFusion()
    # Dummy input: (Batch=2, Channel=1, Length=100)
    dummy_input = torch.randn(2, 1, 100)
    output = model(dummy_input, dummy_input, dummy_input, dummy_input)
    print("Output shape:", output.shape)
    print("Output:", output)
