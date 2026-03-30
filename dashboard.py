import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram, stft, butter, lfilter
import pywt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

# Set Page Config
st.set_page_config(page_title="WESAD Quantum Dashboard", layout="wide")

st.title("🌌 Multimodal Quantum Signal Dashboard")
st.markdown("---")

# -----------------------------------------------------------------------------
# Caching & Loading Logic
# -----------------------------------------------------------------------------
WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'

@st.cache_data
def get_subjects():
    return [s for s in os.listdir(WESAD_PATH) if s.startswith('S')]

@st.cache_data
def load_subject_data(subject_id):
    file_path = os.path.join(WESAD_PATH, subject_id, f"{subject_id}.pkl")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Extract Chest Signals (Original FS = 700Hz)
    chest = data['signal']['chest']
    df = pd.DataFrame({
        'ECG': chest['ECG'].flatten(),
        'EDA': chest['EDA'].flatten(),
        'EMG': chest['EMG'].flatten(),
        'Resp': chest['Resp'].flatten(),
        'Temp': chest['Temp'].flatten()
    })
    return df, data['label'].flatten()

# -----------------------------------------------------------------------------
# Signal Processing Utils
# -----------------------------------------------------------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
subjects = get_subjects()
selected_subject = st.sidebar.selectbox("👤 Select Subject", subjects, index=0)

st.sidebar.markdown("---")
view_range = st.sidebar.slider("📏 View Range (Seconds)", 0, 300, (0, 30))
fs = 700 

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Sharing")
st.sidebar.info("""
**How to share:**
1. **Local Network**: Share the 'Network URL' below.
2. **Global (Temporary)**: Use [Localtunnel](https://theboroer.github.io/localtunnel-www/)
   `npx localtunnel --port 8501`
3. **Permanent**: Deploy to **Streamlit Cloud** via GitHub.
""")

# Load Data
with st.spinner(f"Loading data for {selected_subject}..."):
    df, labels = load_subject_data(selected_subject)

# Slicing
start_idx = int(view_range[0] * fs)
end_idx = int(view_range[1] * fs)
df_slice = df.iloc[start_idx:end_idx].reset_index(drop=True)
time_axis = np.arange(len(df_slice)) / fs

# -----------------------------------------------------------------------------
# TABS: UI Layout
# -----------------------------------------------------------------------------
tab_signals, tab_fft, tab_rhythms, tab_qft = st.tabs([
    "📊 Time Signals", "📈 Time-Frequency (STFT/CWT)", "🧠 Rhythms (Bands)", "⚛️ Quantum (QFT)"
])

with tab_signals:
    st.subheader(f"Multimodal Series: {selected_subject}")
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=("ECG", "EDA", "EMG", "Respiration", "Temperature"))
    colors = ["#FF4B4B", "#00BFFF", "#32CD32", "#FFA500", "#9370DB"]
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=time_axis, y=df_slice[col], name=col, line=dict(color=colors[i])), row=i+1, col=1)
    fig.update_layout(height=800, showlegend=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab_fft:
    st.subheader("Advanced Spectral Analysis")
    mod_tf = st.selectbox("Select Modality for Time-Frequency", df.columns, key="tf_mod")
    y_tf = df_slice[mod_tf].values
    y_tf = y_tf - np.mean(y_tf)
    
    tf_mode = st.radio("Technique", ["Spectrogram (STFT)", "Wavelet (CWT)"], horizontal=True)
    
    if tf_mode == "Spectrogram (STFT)":
        f_s, t_s, Sxx = spectrogram(y_tf, fs, nperseg=256, noverlap=128)
        fig_spec = go.Figure(data=go.Heatmap(x=t_s, y=f_s, z=10*np.log10(Sxx + 1e-10), colorscale='Viridis'))
        fig_spec.update_layout(title="Spectrogram (dB Strength)", xaxis_title="Time (s)", yaxis_title="Freq (Hz)", template="plotly_dark")
        st.plotly_chart(fig_spec, use_container_width=True)
    else:
        scales = np.arange(1, 64)
        coef, freqs = pywt.cwt(y_tf, scales, 'morl', 1/fs)
        fig_wave = go.Figure(data=go.Heatmap(x=time_axis, y=freqs, z=np.abs(coef), colorscale='Magma'))
        fig_wave.update_layout(title="Continuous Wavelet Transform (CWT)", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", template="plotly_dark")
        st.plotly_chart(fig_wave, use_container_width=True)

with tab_rhythms:
    st.subheader("Physiological Rhythms & Frequency Bands")
    st.write("WESAD signals captured in standard frequency windows.")
    mod_rh = st.selectbox("Select Modality to Filter", df.columns, key="rh_mod")
    y_rh = df_slice[mod_rh].values
    
    # Define Bands
    bands = {
        "Delta (0.5-4Hz)": (0.5, 4.0),
        "Theta (4-8Hz)": (4.0, 8.0),
        "Alpha (8-13Hz)": (8.0, 13.0),
        "Beta (13-30Hz)": (13.0, 30.0)
    }
    
    fig_bands = go.Figure()
    band_powers = []
    
    for name, (low, high) in bands.items():
        try:
            filtered = bandpass_filter(y_rh, low, high, fs)
            fig_bands.add_trace(go.Scatter(x=time_axis, y=filtered, name=name))
            band_powers.append(np.sum(filtered**2))
        except:
            st.error(f"Could not filter {name} - check Sampling Rate constraints.")
    
    fig_bands.update_layout(title=f"Filtered Bands for {mod_rh}", xaxis_title="Time (s)", template="plotly_dark", height=500)
    st.plotly_chart(fig_bands, use_container_width=True)
    
    # Power Distribution
    if band_powers:
        fig_pie = go.Figure(data=[go.Pie(labels=list(bands.keys()), values=band_powers, hole=.3)])
        fig_pie.update_layout(title="Relative Power Distribution", template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Quantum Fourier Transform (QFT)
# -----------------------------------------------------------------------------
with tab_qft:
    st.subheader("Quantum Fourier Transform Simulation")
    st.write("Applying QFT to a selected segment of the signal.")
    
    target_modality = st.selectbox("Select Modality for QFT", df.columns, key="qft_mod")
    num_qubits = st.slider("Select Number of Qubits (N)", 3, 6, 4)
    N_qft = 2**num_qubits
    
    st.info(f"Using {N_qft} points for QFT.")
    
    # Selection of segment
    offset = st.slider("Segment Offset (index)", 0, len(df_slice) - N_qft, 0)
    segment = df_slice[target_modality].values[offset : offset + N_qft]
    segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Input Segment (16-64 samples):")
        st.line_chart(segment)
    
    with col2:
        if st.button("🚀 Run Quantum Simulation"):
            # Prepare statevector
            norm = np.linalg.norm(segment)
            statevector = segment / norm if norm > 0 else segment
            
            # Create QFT Circuit
            qc = QuantumCircuit(num_qubits)
            qc.set_statevector(statevector)
            qc.append(QFT(num_qubits), range(num_qubits))
            
            # Simulate
            sim = AerSimulator()
            qc.save_statevector()
            result = sim.run(qc).result()
            final_state = result.get_statevector()
            
            # Probabilities and Phases
            probs = np.abs(final_state)**2
            phases = np.angle(final_state)
            
            # Visualization
            st.write("QFT Output (Probabilities):")
            prob_df = pd.DataFrame({"Prob": probs, "Index": range(len(probs))})
            st.bar_chart(prob_df, x="Index", y="Prob")
            
            st.success("QFT Completed successfully via Qiskit Aer!")
            
    st.markdown("""
    > [!NOTE]
    > **Experimental**: The QFT shows how frequency information is encoded into quantum state amplitudes. 
    > Unlike classical FFT which gives deterministic bin magnitudes, QFT operates on statevectors to enable exponential speedup in specific algorithms.
    """)
