# 🌌 Multimodal Quantum Signal Fusion

A high-performance Physiological Signal Analysis & Stress Detection system designed for the **WESAD** dataset. This project integrates classical Deep Learning architectures with Quantum Fourier Transforms (QFT) and advanced signal processing pipelines.

---

## 🧠 Project Overview

This project implements a robust framework for classifying human stress states (Baseline, Stress, Amusement) using multimodal chest-worn sensors. It bridges the gap between state-of-the-art temporal modeling and quantum-inspired feature fusion.

### 📊 Dataset: WESAD (Chest)
We utilize 5 synchronized modalities recorded at high frequencies, downsampled to **100Hz** for efficient deep learning processing:
1.  **ECG**: Electrocardiogram (Heart activity)
2.  **EDA**: Electrodermal Activity (Skin conductance/Sweat)
3.  **EMG**: Electromyogram (Muscle activity)
4.  **Resp**: Respiration (Breathing patterns)
5.  **Temp**: Skin Temperature

---

## 🛠️ Model Zoo

The system supports 5 powerful architectures, easily toggled via command-line arguments:

| Model | Architecture Description |
| :--- | :--- |
| **`lstm`** | Bidirectional LSTM with sequence mean-pooling for long-term temporal dependencies. |
| **`cnnlstm`** | 1D-CNN for local feature extraction followed by a Bi-LSTM layer. |
| **`transformer`** | Multi-head Self-Attention with Positional Encoding and Global Average Pooling. |
| **`multiscale`** | Parallel 1D-CNNs (Kernels 3, 7, 11) to capture multi-resolution temporal features. |
| **`baseline`** | A complex multi-branch MS-CNN with Temporal Self-Attention for each modality. |

---

## 🚀 Getting Started

### 1. Installation
Ensure you have a Python environment (Conda recommended) with the following dependencies:
```bash
pip install torch qiskit qiskit-aer streamlit plotly pywt scikit-learn seaborn matplotlib
```

### 2. Training (LOSO Framework)
We use a **15-fold Leave-One-Subject-Out (LOSO)** evaluation to ensure cross-subject generalization.

**Run the training pipeline:**
```powershell
# Train the Multi-Scale CNN (Recommended)
python train_advanced.py --model multiscale --window_sec 3 --epochs 50

# Train the Transformer
python train_advanced.py --model transformer --window_sec 3

# Run a quick demo (Fold 0 only)
python train_advanced.py --model cnnlstm --demo
```

### 3. Interactive Quantum Dashboard
Explore signals, analyze frequencies, and simulate quantum transforms in a live web-app.
```powershell
streamlit run dashboard.py
```

**Dashboard Features:**
- **Signals Explorer**: Interactive Plotly charts for all 5 modalities.
- **Time-Frequency**: Spectrograms (STFT) and Wavelet (CWT) Scalograms.
- **Rhythms**: Extraction of Alpha, Beta, Delta, and Theta bands.
- **Preprocessing Lab**: Step-by-step Notch/Bandpass filtering and ICA artifact removal.
- **Quantum QFT**: Real-time Qiskit simulation of the Quantum Fourier Transform.

---

## 📈 Evaluation & Results

Training results, including confusion matrices and LOSO summaries, are automatically saved to:
`output/advanced_loso/<model_name>_<window_size>s/`

**To visualize the cross-subject performance:**
```powershell
python plot_loso_results.py --model multiscale_3s
```

## ⚛️ Quantum Integration
The project features a standalone module for **Quantum Fourier Transforms (QFT)**. By mapping physiological amplitudes to quantum statevectors, we explore high-dimensional frequency encoding that enables exponential speedup in future quantum-hardware deployments.

---

## 📂 File Structure

- `wesad_dataset.py`: Multi-modal data loader and windowing logic.
- `advanced_models.py`: Core PyTorch implementations of all architectures.
- `train_advanced.py`: The 15-fold LOSO training engine.
- `dashboard.py`: Streamlit-based interactive analysis suite.
- `plot_loso_results.py`: Global performance visualization tool.
- `WESAD/`: Raw dataset directory (Expected structure: `S2/S2.pkl`, etc.)
