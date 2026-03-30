import gradio as gr
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram, butter, lfilter
import pywt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'
FS = 700 
MODALITIES = ['ECG', 'EDA', 'EMG', 'Resp', 'Temp']
COLORS = ["#FF4B4B", "#00BFFF", "#32CD32", "#FFA500", "#9370DB"]

def get_subjects():
    if not os.path.exists(WESAD_PATH): return ["WESAD Path Not Found"]
    return [s for s in os.listdir(WESAD_PATH) if s.startswith('S')]

# -----------------------------------------------------------------------------
# Processing Tools
# -----------------------------------------------------------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def load_subject(subject_id):
    file_path = os.path.join(WESAD_PATH, subject_id, f"{subject_id}.pkl")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    chest = data['signal']['chest']
    df = pd.DataFrame({m: chest[m].flatten() for m in MODALITIES})
    return df

# -----------------------------------------------------------------------------
# Chart Generators
# -----------------------------------------------------------------------------
def plot_signals(subject_id, range_sec):
    df = load_subject(subject_id)
    start, end = int(range_sec[0]*FS), int(range_sec[1]*FS)
    slice_df = df.iloc[start:end]
    t = np.arange(len(slice_df)) / FS
    
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=MODALITIES)
    for i, m in enumerate(MODALITIES):
        fig.add_trace(go.Scatter(x=t, y=slice_df[m], name=m, line=dict(color=COLORS[i])), row=i+1, col=1)
    fig.update_layout(height=800, template="plotly_dark", showlegend=False)
    return fig

def plot_tf(subject_id, range_sec, modality, mode):
    df = load_subject(subject_id)
    start, end = int(range_sec[0]*FS), int(range_sec[1]*FS)
    y = df[modality].values[start:end]
    y = y - np.mean(y)
    t = np.arange(len(y)) / FS
    
    if mode == "Spectrogram":
        f_s, t_s, Sxx = spectrogram(y, FS, nperseg=256)
        fig = go.Figure(data=go.Heatmap(x=t_s, y=f_s, z=10*np.log10(Sxx+1e-10), colorscale='Viridis'))
        fig.update_layout(title=f"Spectrogram: {modality}", yaxis_title="Hz", template="plotly_dark")
    else:
        scales = np.arange(1, 64)
        coef, freqs = pywt.cwt(y, scales, 'morl', 1/FS)
        fig = go.Figure(data=go.Heatmap(x=t, y=freqs, z=np.abs(coef), colorscale='Magma'))
        fig.update_layout(title=f"Wavelet Scalogram: {modality}", yaxis_title="Hz", template="plotly_dark")
    return fig

def plot_bands(subject_id, range_sec, modality):
    df = load_subject(subject_id)
    start, end = int(range_sec[0]*FS), int(range_sec[1]*FS)
    y = df[modality].values[start:end]
    t = np.arange(len(y)) / FS
    
    bands = {"Delta": (0.5, 4.0), "Theta": (4.0, 8.0), "Alpha": (8.0, 13.0), "Beta": (13.0, 30.0)}
    fig = go.Figure()
    powers = []
    
    for name, (low, high) in bands.items():
        filt = bandpass_filter(y, low, high, FS)
        fig.add_trace(go.Scatter(x=t, y=filt, name=name))
        powers.append(np.sum(filt**2))
    
    fig.update_layout(title="Extracted Physiological Rhythms", template="plotly_dark")
    
    fig_pie = go.Figure(data=[go.Pie(labels=list(bands.keys()), values=powers, hole=.3)])
    fig_pie.update_layout(title="Power Distribution", template="plotly_dark")
    return fig, fig_pie

def run_qft(subject_id, range_sec, modality, num_qubits):
    df = load_subject(subject_id)
    n_pts = 2**num_qubits
    start = int(range_sec[0]*FS)
    y = df[modality].values[start:start+n_pts]
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    
    # State Preparation
    norm = np.linalg.norm(y)
    sv = y / norm if norm > 0 else y
    
    qc = QuantumCircuit(num_qubits)
    qc.set_statevector(sv)
    qc.append(QFT(num_qubits), range(num_qubits))
    
    sim = AerSimulator()
    qc.save_statevector()
    final_sv = sim.run(qc).result().get_statevector()
    
    probs = np.abs(final_sv)**2
    fig = go.Figure(data=[go.Bar(x=[f"|{i}>" for i in range(len(probs))], y=probs, marker_color='#00BFFF')])
    fig.update_layout(title=f"QFT Probability Statevector (N={num_qubits})", template="plotly_dark")
    return fig

# -----------------------------------------------------------------------------
# Gradio UI Layout
# -----------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Multimodal Quantum Portal") as demo:
    gr.Markdown("# 🌌 Multimodal Quantum Portal (Gradio)")
    gr.Markdown("Directly share this dashboard with public links including QFT and Wavelet analysis.")
    
    with gr.Row():
        subj_inp = gr.Dropdown(choices=get_subjects(), label="Select Subject", value="S2")
        range_inp = gr.Slider(0, 300, value=(0, 30), step=1, label="Time Window (Seconds)")
    
    with gr.Tabs():
        with gr.Tab("📊 Signals View"):
            sig_btn = gr.Button("Fetch Multimodal Signals", variant="primary")
            sig_plot = gr.Plot()
            sig_btn.click(plot_signals, inputs=[subj_inp, range_inp], outputs=sig_plot)
            
        with gr.Tab("⚡ Time-Frequency"):
            with gr.Row():
                mod_tf = gr.Dropdown(MODALITIES, label="Modality", value="ECG")
                mode_tf = gr.Radio(["Spectrogram", "Wavelet"], label="Method", value="Spectrogram")
            tf_btn = gr.Button("Generate Spectrum", variant="primary")
            tf_plot = gr.Plot()
            tf_btn.click(plot_tf, inputs=[subj_inp, range_inp, mod_tf, mode_tf], outputs=tf_plot)
            
        with gr.Tab("🧠 Rhythms (Bands)"):
            mod_rh = gr.Dropdown(MODALITIES, label="Modality", value="EDA")
            rh_btn = gr.Button("Extract Bands", variant="primary")
            with gr.Row():
                rh_plot = gr.Plot()
                rh_pie = gr.Plot()
            rh_btn.click(plot_bands, inputs=[subj_inp, range_inp, mod_rh], outputs=[rh_plot, rh_pie])
            
        with gr.Tab("⚛️ Quantum QFT"):
            with gr.Row():
                mod_q = gr.Dropdown(MODALITIES, label="Modality", value="ECG")
                qubits = gr.Slider(3, 6, value=4, step=1, label="Qubits (State Points = 2^N)")
            q_btn = gr.Button("Run QFT Simulation", variant="primary")
            q_plot = gr.Plot()
            q_btn.click(run_qft, inputs=[subj_inp, range_inp, mod_q, qubits], outputs=q_plot)

# -----------------------------------------------------------------------------
# Launch with Public Sharing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # set share=True to generate a public world-wide URL
    demo.launch(share=True)
