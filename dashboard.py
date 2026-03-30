import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram, stft, butter, lfilter, iirnotch
import pywt
from sklearn.decomposition import FastICA
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

def notch_filter(data, cutoff, fs, q=30):
    nyq = 0.5 * fs
    freq = cutoff / nyq
    b, a = iirnotch(freq, q)
    return lfilter(b, a, data)

def normalize_signal(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

def get_state_intervals(slice_labels, start_time_sec, fs):
    """Detects continuous intervals of states for Plotly vrect shading."""
    intervals = []
    if len(slice_labels) == 0: return intervals
    
    curr_label = slice_labels[0]
    curr_start = 0
    
    for i in range(1, len(slice_labels)):
        if slice_labels[i] != curr_label:
            intervals.append((
                start_time_sec + curr_start/fs,
                start_time_sec + i/fs,
                curr_label
            ))
            curr_label = slice_labels[i]
            curr_start = i
            
    intervals.append((
        start_time_sec + curr_start/fs,
        start_time_sec + len(slice_labels)/fs,
        curr_label
    ))
    return intervals

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
subjects = get_subjects()
selected_subject = st.sidebar.selectbox("👤 Select Subject", subjects, index=0)

st.sidebar.markdown("---")
view_range = st.sidebar.slider("📏 View Range (Seconds)", 0, 300, (0, 30))
fs = 700 

st.sidebar.markdown("---")
show_overlay = st.sidebar.toggle("🔴 Show Normal/Abnormal Overlay", value=True)
st.sidebar.markdown("""
- 🟢 **Normal**: Baseline / Amusement
- 🔴 **Abnormal**: Stress
- ⚪ **Transient**: Transition / Other
""")

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
labels_slice = labels[start_idx:end_idx]
time_axis = np.arange(len(df_slice)) / fs + view_range[0]

# -----------------------------------------------------------------------------
# TABS: UI Layout
# -----------------------------------------------------------------------------
tab_signals, tab_compare, tab_fft, tab_rhythms, tab_preprocess, tab_qft = st.tabs([
    "📊 Time Signals", "⚖️ Compare States", "📈 Time-Frequency", "🧠 Rhythms (Bands)", "🛠️ Preprocessing Lab", "⚛️ Quantum (QFT)"
])

# Label -> Category Helper
def get_label_meta(label):
    if label in [1, 3]: return "Normal", "rgba(0, 255, 0, 0.15)" # Green
    if label == 2: return "Abnormal", "rgba(255, 0, 0, 0.15)" # Red
    return "Transient", "rgba(128, 128, 128, 0.1)" # Grey

with tab_signals:
    st.subheader(f"Multimodal Series: {selected_subject}")
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=("ECG", "EDA", "EMG", "Respiration", "Temperature"))
    colors = ["#FF4B4B", "#00BFFF", "#32CD32", "#FFA500", "#9370DB"]
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=time_axis, y=df_slice[col], name=col, line=dict(color=colors[i])), row=i+1, col=1)
    
    if show_overlay:
        intervals = get_state_intervals(labels_slice, view_range[0], fs)
        for start, end, lbl in intervals:
            status, color = get_label_meta(lbl)
            fig.add_vrect(x0=start, x1=end, fillcolor=color, layer="below", line_width=0, annotation_text=status if i==0 else None)
            
    fig.update_layout(height=800, showlegend=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab_compare:
    st.subheader("⚖️ Normal vs Abnormal Physiological Comparison")
    st.markdown("Direct side-by-side comparison of 10-second segments from each state.")
    
    mod_c = st.selectbox("Select Modality to Compare", df.columns, key="comp_mod")
    
    # Extract Normal (Baseline/Amusement: 1, 3)
    idx_norm = np.where(np.isin(labels, [1, 3]))[0]
    # Extract Abnormal (Stress: 2)
    idx_abnorm = np.where(labels == 2)[0]
    
    if len(idx_norm) > 7000 and len(idx_abnorm) > 7000:
        win = 7000 # 10 seconds
        s_norm = df.iloc[idx_norm[0]:idx_norm[0]+win][mod_c].values
        s_abnorm = df.iloc[idx_abnorm[0]:idx_abnorm[0]+win][mod_c].values
        t_comp = np.arange(win) / fs
        
        # Determine shared Y-axis (Crucial for visual comparison)
        y_min = min(np.min(s_norm), np.min(s_abnorm))
        y_max = max(np.max(s_norm), np.max(s_abnorm))
        y_range = [y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min)]

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Normal (Baseline)")
            f1 = go.Figure()
            f1.add_trace(go.Scatter(x=t_comp, y=s_norm, line=dict(color="#00FF00")))
            f1.update_layout(height=400, template="plotly_dark", yaxis_range=y_range, xaxis_title="Seconds")
            st.plotly_chart(f1, use_container_width=True)
            
        with col2:
            st.markdown("### 🔴 Abnormal (Stress)")
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=t_comp, y=s_abnorm, line=dict(color="#FF0000")))
            f2.update_layout(height=400, template="plotly_dark", yaxis_range=y_range, xaxis_title="Seconds")
            st.plotly_chart(f2, use_container_width=True)
            
        # Comparison Table
        st.write("### Statistical Benchmarking")
        m_norm = np.mean(s_norm)
        m_abnorm = np.mean(s_abnorm)
        diff = ((m_abnorm - m_norm) / (abs(m_norm) + 1e-8)) * 100
        
        st.table(pd.DataFrame({
            "Metric": ["Average Value", "Signal Variance (Std)", "State Status"],
            "Normal (Green)": [f"{m_norm:.4f}", f"{np.std(s_norm):.4f}", "Resting/Baseline"],
            "Abnormal (Red)": [f"{m_abnorm:.4f}", f"{np.std(s_abnorm):.4f}", "High Stress/Arousal"],
            "Difference (%)": [f"{diff:+.1f}%", f"{((np.std(s_abnorm)/np.std(s_norm))-1)*100:+.1f}%", "---"]
        }))
    else:
        st.warning("Insufficient data regions found for this subject to perform a 10s comparison.")

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
            st.error(f"Could not filter {name}")
    
    if show_overlay:
        intervals = get_state_intervals(labels_slice, view_range[0], fs)
        for start, end, lbl in intervals:
            _, color = get_label_meta(lbl)
            fig_bands.add_vrect(x0=start, x1=end, fillcolor=color, layer="below", line_width=0)
            
    fig_bands.update_layout(title=f"Filtered Bands for {mod_rh}", xaxis_title="Time (s)", template="plotly_dark", height=500)
    st.plotly_chart(fig_bands, use_container_width=True)
    
    if band_powers:
        fig_pie = go.Figure(data=[go.Pie(labels=list(bands.keys()), values=band_powers, hole=.3)])
        fig_pie.update_layout(title="Relative Power Distribution", template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

with tab_preprocess:
    st.subheader("🛠️ Step-by-Step Signal Preprocessing Lab")
    st.markdown("Visualize how the raw physiological data is cleaned and normalized.")
    
    mod_p = st.selectbox("Select Modality to Clean", df_slice.columns, key="p_mod")
    y_raw = df_slice[mod_p].values
    
    col_steps, col_plots = st.columns([1, 4])
    
    with col_steps:
        st.write("**Controls**")
        notch_val = st.slider("Notch Filter (Hz)", 40, 70, 60)
        low_cut = st.slider("Bandpass Low (Hz)", 0.1, 5.0, 0.5)
        high_cut = st.slider("Bandpass High (Hz)", 10, 100, 40)
        do_ica = st.checkbox("Apply Artifact Removal (ICA)", value=True)
    
    with col_plots:
        def plot_step(y, title):
            f = go.Figure()
            f.add_trace(go.Scatter(x=time_axis, y=y, name="Signal", line=dict(color="#00BFFF")))
            if show_overlay:
                intervals = get_state_intervals(labels_slice, view_range[0], fs)
                for start, end, lbl in intervals:
                    _, color = get_label_meta(lbl)
                    f.add_vrect(x0=start, x1=end, fillcolor=color, layer="below", line_width=0)
            f.update_layout(title=title, height=250, margin=dict(t=30, b=30), template="plotly_dark", xaxis_visible=False)
            return f

        # Step 1: Raw
        st.plotly_chart(plot_step(y_raw, "1. Raw Signal"), use_container_width=True)
        
        # Step 2: Filtering
        y_notched = notch_filter(y_raw, notch_val, fs)
        y_filtered = bandpass_filter(y_notched, low_cut, high_cut, fs)
        st.plotly_chart(plot_step(y_filtered, f"2. Filtered (Notch {notch_val}Hz + Bandpass)"), use_container_width=True)
        
        # Step 3: Artifact Removal (ICA)
        if do_ica:
            ica_data = df_slice.values
            ica = FastICA(n_components=5, random_state=42)
            S_ = ica.fit_transform(ica_data)
            corrs = [np.corrcoef(S_[:, i], y_filtered)[0, 1] for i in range(5)]
            best_comp = np.argmax(np.abs(corrs))
            y_cleaned = S_[:, best_comp]
            st.plotly_chart(plot_step(y_cleaned, "3. Artifact Cleaned (ICA)"), use_container_width=True)
        else:
            y_cleaned = y_filtered
            
        # Step 4: Normalization
        y_norm = normalize_signal(y_cleaned)
        f_norm = plot_step(y_norm, "4. Normalized (0-1 Scaling)")
        f_norm.update_layout(xaxis_visible=True, xaxis_title="Time (s)")
        st.plotly_chart(f_norm, use_container_width=True)

with tab_qft:
    st.subheader("Quantum Fourier Transform Simulation")
    st.write("Applying QFT to a selected segment of the signal.")
    
    target_modality = st.selectbox("Select Modality for QFT", df.columns, key="qft_mod")
    num_qubits = st.slider("Select Number of Qubits (N)", 3, 6, 4)
    N_qft = 2**num_qubits
    
    st.info(f"Using {N_qft} points for QFT.")
    
    offset = st.slider("Segment Offset (index)", 0, len(df_slice) - N_qft, 0)
    segment = df_slice[target_modality].values[offset : offset + N_qft]
    segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Input Segment (16-64 samples):")
        st.line_chart(segment)
    
    with col2:
        if st.button("🚀 Run Quantum Simulation"):
            norm = np.linalg.norm(segment)
            statevector = segment / norm if norm > 0 else segment
            qc = QuantumCircuit(num_qubits)
            qc.set_statevector(statevector)
            qc.append(QFT(num_qubits), range(num_qubits))
            sim = AerSimulator()
            qc.save_statevector()
            result = sim.run(qc).result()
            final_state = result.get_statevector()
            probs = np.abs(final_state)**2
            st.write("QFT Output (Probabilities):")
            prob_df = pd.DataFrame({"Prob": probs, "Index": range(len(probs))})
            st.bar_chart(prob_df, x="Index", y="Prob")
            st.success("QFT Completed successfully via Qiskit Aer!")
            
    st.markdown("""
    > [!NOTE]
    > **Experimental**: The QFT shows how frequency information is encoded into quantum state amplitudes. 
    """)
