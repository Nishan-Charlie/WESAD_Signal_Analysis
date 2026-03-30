import pickle
import os

data_path = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD\S2\S2.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

print("Keys in pickle file:", data.keys())
print("\nSignal keys:", data['signal'].keys())
print("\nChest signal keys:", data['signal']['chest'].keys())
print("\nWrist signal keys:", data['signal']['wrist'].keys())

# Check shapes of some signals
print("\nShapes:")
print("ECG:", data['signal']['chest']['ECG'].shape)
print("EDA (chest):", data['signal']['chest']['EDA'].shape)
print("EMG:", data['signal']['chest']['EMG'].shape)
print("Resp:", data['signal']['chest']['Resp'].shape)
print("Label:", data['label'].shape)

import numpy as np
unique_labels, counts = np.unique(data['label'], return_counts=True)
print("\nLabels and counts:", dict(zip(unique_labels, counts)))
