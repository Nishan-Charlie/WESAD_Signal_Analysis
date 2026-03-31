# 🌌 Multimodal Quantum Signal Fusion

A high-performance Physiological Signal Analysis & Stress Detection system designed for the **WESAD** dataset. This project integrates classical Deep Learning architectures with Quantum Fourier Transforms (QFT) and PennyLane-backed Hybrid Quantum-Classical pipelines.

---

## 🧠 Project Overview

This project implements a robust framework for classifying human stress states (Baseline, Stress, Amusement) using multimodal chest-worn sensors. It bridges the gap between state-of-the-art temporal modeling, hardware-optimal scaling (Flash/Linear Attention), and quantum-inspired feature fusion.

### 📊 Dataset: WESAD (Chest)
We utilize 5 synchronized modalities recorded at high frequencies, downsampled to **100Hz** for efficient deep learning processing:
1.  **ECG**: Electrocardiogram (Heart activity)
2.  **EDA**: Electrodermal Activity (Skin conductance/Sweat)
3.  **EMG**: Electromyogram (Muscle activity)
4.  **Resp**: Respiration (Breathing patterns)
5.  **Temp**: Skin Temperature

---

## 🛠️ Model Zoo

The system supports 9 powerful architectures, seamlessly toggled via command-line arguments:

| Model | Architecture Description |
| :--- | :--- |
| **`lstm`** | Bidirectional LSTM with sequence mean-pooling for long-term temporal dependencies. |
| **`cnnlstm`** | 1D-CNN (with SE-Blocks) for local feature extraction followed by a Bi-LSTM layer. |
| **`transformer`** | Multi-head Self-Attention with Positional Encoding and Global Average Pooling. Supports Exact (`flash`), Subquadratic (`linear`), and standard attentions. |
| **`multiscale`** | Parallel 1D-CNNs (Kernels 3, 7, 11) dynamically recalibrated via an autotuning Squeeze-and-Excitation (SE-Block) integration. |
| **`baseline`** | A complex multi-branch MS-CNN with Temporal Self-Attention for each modality. |
| **`*-quantum`** | Four hybrid variants (e.g. `transformer-quantum`) that route the final embeddings through a parameterized PennyLane quantum circuit configured with Data Re-uploading before final classification. Optimized via von Neumann entropy regularization. |

---

## 🚀 Getting Started

### 1. Installation
Ensure you have a Python environment (Conda `quantum_fusion` recommended) with the following dependencies:
```bash
pip install torch "pennylane<0.43.0" "autoray<0.8.0" qiskit qiskit-aer streamlit gradio plotly pywt scikit-learn seaborn matplotlib
```
*(Note: PennyLane and Autoray specific versions are required to avoid namespace conflicts on standard setups).*

### 2. Training (LOSO Framework)
We use a **15-fold Leave-One-Subject-Out (LOSO)** evaluation to ensure cross-subject generalization.

**Run the training pipeline:**
```powershell
# Train the Multi-Scale CNN (Recommended)
python train_advanced.py --model multiscale --window_sec 3 --epochs 50

# Train the Classical Transformer with O(N) Linear Attention
python train_advanced.py --model transformer --attn_type linear --window_sec 3

# Train a Hybrid Quantum Architecture with exact Flash Attention and Entropy Regularization
python train_advanced.py --model transformer-quantum --attn_type flash --lambda_entropy 0.1

# Run a quick demo (Fold 0 only)
python train_advanced.py --model cnn-lstm-quantum --demo
```

### 3. Interactive Dashboards
Explore signals, analyze frequencies, and simulate quantum transforms in our live web-apps:
```powershell
streamlit run dashboard.py
python gradio_app.py
```

**Dashboard Features:**
- **Signals Explorer**: Interactive Plotly charts for all modalities.
- **Time-Frequency**: Spectrograms (STFT) and Wavelet (CWT) Scalograms.
- **Preprocessing Lab & Rhythms**: Step-by-step filtering, ICA, and brain/heart rhythm extraction.
- **Quantum QFT (Qiskit)**: Statevector visualizations mapped from raw physiological scaling.

---

## 📈 Evaluation & Results

Training results, including confusion matrices, history arrays, and comprehensive metrics, are automatically saved to:
`output/advanced_loso/<model_name>_<window_size>s/fold_X/`

**To visualize the cross-subject performance:**
```powershell
python visualization/plot_loso_results.py
```

---

## 📂 File Structure

The project has been streamlined for active research workflows:

- `train_advanced.py`: The 15-fold LOSO model training engine.
- `advanced_models.py`: Core PyTorch mechanisms (includes Squeeze/Excitation & Native Attentions) alongside PennyLane Quantum backends.
- `wesad_dataset.py`: Multi-modal data loader and preprocessing logic.
- `dashboard.py` & `gradio_app.py`: Web-based interactive execution suites.
- `WESAD/`: Raw dataset directory.
- `preprocessing/`: Scripts to check dataset balance, missing IDs, and sequence distributions.
- `visualization/`: Aggregation plotting scripts handling JSON compilation from the results dir.
- `legacy/`: Pre-advanced models and basic scripts.
