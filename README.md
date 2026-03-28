# Multimodal Quantum Fusion

A Hybrid Quantum-Classical Neural Network designed to fuse multimodal physiological data (ECG, EDA, EMG, Resp) for state classification (Baseline, Stress, Amusement).

## 🧠 Project Overview

This project implements a novel architecture that combines the feature extraction power of classical Deep Learning with the high-dimensional fusion capabilities of Quantum Computing.

### Key Components

1.  **Independent Feature Extraction (Classical)**:
    - **Input**: 4 Channels (ECG, EDA, EMG, Resp).
    - **Architecture**: Each channel is processed by a dedicated **1D-CNN + LSTM** branch to extract temporal features.
    - **Output**: 32-dimensional feature vector per channel.
2.  **Quantum Projection (Bottleneck)**:
    - The concatenated features (128 total) are compressed into **8 quantum-ready features** via a classical Dense Layer.
3.  **8-Qubit Quantum Fusion Layer**:
    - **Encoding**: Data is embedded into the quantum state using **Amplitude Encoding** (via RX gates).
    - **Entanglement**: A custom **Strongly Entangling Layer** structure with increasing connectivity depth (Nearest Neighbor $\rightarrow$ Distant $\rightarrow$ Global) allows for complex feature interaction.
    - **Measurement**: Expectation values of Pauli-Z operators are measured for all 8 qubits.
4.  **Classification**:
    - A final classical Linear layer maps the 8 quantum outputs to the 3 target classes.

## 📂 File Structure

```
MultiModal_Quantum_Fusion/
├── model.py           # Contains the Hybrid Model Architecture (ClassicalBranch + Fusion)
├── train.py           # Training script with dummy data generation and training loop
├── requirements.txt   # Python dependencies
└── README.md          # Project Documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PennyLane
- Target Device: CPU or GPU (CUDA)

### Installation

1.  Clone the repository or download the files.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

To train the model (currently set up with dummy data for verification):

```bash
python train.py
```

The script will:

1.  Initialize the hybrid model.
2.  Generate random dummy data for 4 channels.
3.  Run a training loop for 5 epochs.
4.  Output the Loss and Accuracy per epoch.

## 🛠️ Model Architecture Details

### Classical Branch

- **Conv1d**: Kernel size 3, ReLU activation, MaxPool.
- **LSTM**: Captures sequence dependencies from the CNN output.

### Quantum Circuit (PennyLane)

- **Qubits**: 8.
- **Depth**: 3 Layers.
- **Ansatz**:
  - _Rotations_: Trainable parameters.
  - _CNOTs_: Entangles qubits with varying strides (1, 2, 3) to ensure global information flow.

## 📊 Classes

- 0: Baseline
- 1: Stress
- 2: Amusement
"# WESAD_Psychological-Analysis" 
"# WESAD_Psychological-Analysis" 
"# WESAD_Signal_Analysis" 
