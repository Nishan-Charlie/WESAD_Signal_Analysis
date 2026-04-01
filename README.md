# 🌌 Multimodal Quantum Signal Fusion

A state-of-the-art **Universal Multimodal Research Framework** for Physiological Signal Analysis & Stress Detection on the **WESAD** dataset. This project integrates cutting-edge Deep Learning architectures with high-performance PennyLane Quantum backends.

---

## 🧠 Project Core: Universal Multimodal Architecture

This framework implements a modular **Backbone + Fusion + Backend** paradigm. This decoupling allows researchers to conduct exhaustive ablation studies across temporal modeling techniques, signal integration strategies, and quantum-classical hybridity.

### 📊 Dataset: WESAD (Chest-Worn)
We process 5 high-fidelity synchronized modalities (ECG, EDA, EMG, Resp, Temp) downsampled to **100Hz** for balanced computational efficiency and feature resolution.

---

## 🛠️ Model Research Zoo

### 🧩 1. Multimodal Fusion Strategies (`--fusion`)
Decide how modality information is integrated into the final decision:

| Fusion Strategy | Description | Research Rationale |
| :--- | :--- | :--- |
| **`early`** | Concatenates raw signals before the backbone. | Analyzes low-level cross-modal correlations. |
| **`mid`** | Independent backbone branches per modality. | Captures modality-specific "temporal languages" before fusion. |
| **`late`** | Independent models with decision-level averaging. | Robust to sensor failure; captures independent modality expertize. |

### 🚀 2. Temporal Backbones (`--backbone`)
Choose the underlying architecture for feature extraction:

- **`lstm`**: Bidirectional LSTM with Global Mean Pooling for long-term physiological trends.
- **`cnn`**: Multi-Scale 1D-CNN (Kernels 3, 7, 11) using **SE-Block** channel recalibration.
- **`transformer`**: Multi-head Self-Attention with Positional Encoding for global dependency modeling.
- **`cnnlstm`**: Hybrid architecture utilizing CNNs for local spatial features and LSTMs for global sequence dynamics.

### ⚡ 3. Quantum-Classical Hybrid Backend (`--quantum`)
The system features a high-performance quantum classifier optimized for research-scale execution:
- **10 Qubits / 4 Layers**: High-dimensional feature mapping via Data Re-uploading.
- **`lightning.qubit` Device**: C++ optimized state-vector simulation ($10\times$ speedup).
- **Adjoint Differentiation**: $O(P)$ memory-efficient gradient calculation for deep hybrid circuits.
- **Entropic Regularization**: Minimizes von Neumann entropy to encourage clear state separations.

---

## 🚀 Execution Guide

### 1. Installation
```bash
pip install torch pennylane-lightning "pennylane<0.43.0" "autoray<0.8.0" qiskit qiskit-aer streamlit scikit-learn tqdm
```

### 2. Running a Research Ablation Suite
The results are automatically organized by their configuration for easy comparison (e.g., `output/advanced_loso/lstm_mid_quantum_30s/`).

**Classical Baselines:**
```powershell
python train_advanced.py --backbone cnn --fusion mid --epochs 50
python train_advanced.py --backbone lstm --fusion early --epochs 50
```

**Quantum Hybrid Experiments:**
```powershell
python train_advanced.py --backbone transformer --fusion mid --quantum --epochs 30
python train_advanced.py --backbone cnnlstm --fusion early --quantum --epochs 30
```

---

## 📂 Project Architecture
```text
├── train_advanced.py       # Unified 15-fold LOSO training engine
├── advanced_models.py      # UniversalMultimodalModel & Quantum Backends
├── wesad_dataset.py        # Multi-modal data loader & Augmentations
├── dashboard.py            # Streamlit Signal Exploration Suite
└── output/                 # Automatically categorized research results
```

---

## 📈 Evaluation Metrics
For every experiment, the system automatically generates:
- **LOSO Accuracy & F1-Macro** (Cross-subject generalization)
- **Confusion Matrices** (Detailed per-class performance)
- **Training History** (Loss and Accuracy curves)
- **Quantum Entanglement Logs** (If using the Quantum backend)
