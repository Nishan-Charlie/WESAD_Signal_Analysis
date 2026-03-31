import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml


class LSTMModel(nn.Module):
    def __init__(self, num_features=5, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        # Bi-LSTM
        # Input shape: (Batch, Seq, Features)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Head
        # Bidirectional means hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq, Features)
        lstm_out, _ = self.lstm(x)
        
        # Mean pooling over all timesteps — much better than last-state for long sequences
        # as it aggregates information across the entire window rather than relying
        # on a single endpoint that suffers from vanishing gradients
        pooled = lstm_out.mean(dim=1)  # (Batch, Hidden*2)
        
        logits = self.classifier(pooled)
        return logits


class CNNLSTMModel(nn.Module):
    def __init__(self, num_features=5, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super(CNNLSTMModel, self).__init__()
        
        # CNN Block
        # Conv1D expects (Batch, Channels, Seq)
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM Block
        # After CNN, channels = 128. This becomes the input_size for LSTM.
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # User specified 0.4 for FC in CNN-LSTM
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq, Features)
        # Permute for CNN: (Batch, Features, Seq)
        x = x.permute(0, 2, 1)
        
        x = self.cnn(x)
        
        # Permute back for LSTM: (Batch, Seq_reduced, Channels=128)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        logits = self.classifier(last_out)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Seq, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_features=5, d_model=64, nhead=8, num_layers=3, dim_feedforward=256, num_classes=3, dropout=0.1, attn_type='flash'):
        super(TransformerModel, self).__init__()
        
        # 1. Input Projection
        self.embedding = nn.Linear(num_features, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder
        self.attn_type = attn_type
        if attn_type == 'linear':
            self.layers = nn.ModuleList([FastLinearAttentionLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
            self.transformer_encoder = None
        elif attn_type == 'flash':
            self.layers = nn.ModuleList([FastTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
            self.transformer_encoder = None
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.layers = None
        
        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq, Features)
        
        # Project inputs to d_model
        x = self.embedding(x) # (Batch, Seq, d_model)
        
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        if self.layers is not None:
            for layer in self.layers:
                x = layer(x)
            x_enc = x
        else:
            x_enc = self.transformer_encoder(x) # (Batch, Seq, d_model)
        
        # Global Average Pooling across the sequence dimension
        x = x_enc.mean(dim=1) # (Batch, d_model)
        
        logits = self.classifier(x)
        return logits


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduced = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class FastTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super(FastTransformerEncoderLayer, self).__init__()
        self.nhead = nhead
        self.d_model = d_model
        assert d_model % nhead == 0
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
    def forward(self, src):
        B, S, E = src.shape
        q = self.q_proj(src).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(src).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, E)
        attn_out = self.out_proj(attn_out)
        
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        ff_out = self.ff(src)
        src = src + self.dropout2(ff_out)
        return self.norm2(src)

class FastLinearAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super(FastLinearAttentionLayer, self).__init__()
        self.nhead = nhead
        self.d_model = d_model
        assert d_model % nhead == 0
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
    def _elu_map(self, x):
        return F.elu(x) + 1.0

    def forward(self, src):
        B, S, E = src.shape
        q = self.q_proj(src).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(src).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        # Apply kernel feature map
        q_m = self._elu_map(q)
        k_m = self._elu_map(k)
        
        # O(N) evaluation strategy
        kv = torch.matmul(k_m.transpose(-2, -1), v)
        z = 1 / (torch.matmul(q_m, k_m.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
        
        attn_out = torch.matmul(q_m, kv) * z
        attn_out = attn_out.transpose(1, 2).reshape(B, S, E)
        attn_out = self.out_proj(attn_out)
        
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        ff_out = self.ff(src)
        src = src + self.dropout2(ff_out)
        return self.norm2(src)

class MultiScale1DCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super(MultiScale1DCNN, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv11 = nn.Conv1d(in_channels, out_channels, kernel_size=11, padding=5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        c3 = self.relu(self.conv3(x))
        c7 = self.relu(self.conv7(x))
        c11 = self.relu(self.conv11(x))
        return torch.cat([c3, c7, c11], dim=1)


class TemporalAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(TemporalAttention1D, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (Batch, Channels, Seq)
        x_permuted = x.permute(0, 2, 1) # (Batch, Seq, Channels)
        attn_out, _ = self.mha(x_permuted, x_permuted, x_permuted)
        out = self.layer_norm(x_permuted + attn_out)
        return out.permute(0, 2, 1)


class ModalityBranch(nn.Module):
    def __init__(self, in_channels=1, cnn_out_channels=16, num_heads=4):
        super(ModalityBranch, self).__init__()
        self.ms_cnn = MultiScale1DCNN(in_channels, cnn_out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.attention = TemporalAttention1D(embed_dim=cnn_out_channels * 3, num_heads=num_heads)
        
    def forward(self, x):
        # x: (Batch, 1, Seq)
        x = self.ms_cnn(x)
        x = self.pool(x)
        x = self.attention(x)
        return x


class ClassicalBaseline(nn.Module):
    def __init__(self, num_features=5, latent_dim=8, num_classes=3):
        """
        Original Temporal-Attention CNN from classical_baseline.py
        Adapted for 5 multimodal channels.
        """
        super(ClassicalBaseline, self).__init__()
        cnn_out_channels = 16
        
        # 5 branches for 5 modalities (ECG, EDA, EMG, Resp, Temp)
        self.branch_ecg = ModalityBranch(1, cnn_out_channels)
        self.branch_eda = ModalityBranch(1, cnn_out_channels)
        self.branch_emg = ModalityBranch(1, cnn_out_channels)
        self.branch_resp = ModalityBranch(1, cnn_out_channels)
        self.branch_temp = ModalityBranch(1, cnn_out_channels)
        
        fused_channels = 5 * (cnn_out_channels * 3) # 240 features
        
        self.dim_reduction = nn.Sequential(
            nn.Conv1d(fused_channels, latent_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq, 5)
        # Permute to (Batch, 5, Seq)
        x = x.permute(0, 2, 1)
        
        # Split modalities for branch processing
        # (Batch, 1, Seq)
        f_ecg = self.branch_ecg(x[:, 0:1, :])
        f_eda = self.branch_eda(x[:, 1:2, :])
        f_emg = self.branch_emg(x[:, 2:3, :])
        f_resp = self.branch_resp(x[:, 3:4, :])
        f_temp = self.branch_temp(x[:, 4:5, :])
        
        fused = torch.cat([f_ecg, f_eda, f_emg, f_resp, f_temp], dim=1)
        
        latent = self.dim_reduction(fused).squeeze(-1) # (Batch, latent_dim)
        
        logits = self.classifier(latent)
        return logits


class MultiScaleCNNModel(nn.Module):
    def __init__(self, num_features=5, num_classes=3, out_channels=32, dropout=0.3):
        """
        A unified Multi-Scale 1D CNN that processes all modalities in parallel.
        More compact than ClassicalBaseline but keeps the multi-resolution benefits.
        """
        super(MultiScaleCNNModel, self).__init__()
        
        # Parallel Multi-Scale Blocks
        # (Batch, Channels=5, Seq)
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_features, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(num_features, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv1d(num_features, out_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # Concatenated features -> 3 * out_channels
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq, Features) -> (Batch, Features, Seq)
        x = x.permute(0, 2, 1)
        
        c3 = self.conv3(x)
        c7 = self.conv7(x)
        c11 = self.conv11(x)
        
        fused = torch.cat([c3, c7, c11], dim=1)
        logits = self.classifier(fused)
        return logits


class NoiseAwareAugmentation(nn.Module):
    def __init__(self, noise_std=0.01, bw_amplitude=0.05):
        super(NoiseAwareAugmentation, self).__init__()
        self.noise_std = noise_std
        self.bw_amplitude = bw_amplitude
        
    def forward(self, x):
        if not self.training:
            return x
            
        # x is assumed to be (Batch, Seq, Features)
        if x.dim() == 3:
            batch, seq, features = x.shape
            
            # Add Gaussian noise
            noise = torch.randn_like(x) * self.noise_std
            
            # Add baseline wander (low freq sine)
            t = torch.arange(seq, dtype=x.dtype, device=x.device) / seq
            cycles = torch.empty(batch, 1, features, device=x.device).uniform_(1, 5)
            phase = torch.empty(batch, 1, features, device=x.device).uniform_(0, 2*np.pi)
            
            t_expand = t.unsqueeze(0).unsqueeze(2)
            bw = self.bw_amplitude * torch.sin(2 * np.pi * cycles * t_expand + phase)
            
            return x + noise + bw
        return x

class HybridLoss(nn.Module):
    def __init__(self, lambda_entropy=0.1):
        super(HybridLoss, self).__init__()
        self.lambda_entropy = lambda_entropy
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits, entropies, targets):
        ce = self.ce_loss(logits, targets)
        # Minimize entropy: penalize large entropy to encourage clear states
        entropy_loss = entropies.mean()
        return ce + self.lambda_entropy * entropy_loss

class MultiScaleModalityBlock(nn.Module):
    def __init__(self, in_channels=1, out_features=8):
        super(MultiScaleModalityBlock, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # Concatenate: 16+16+16 = 48 -> SEBlock -> AdaptiveAvgPool1d(1) -> Linear -> 8
        self.se = SEBlock1D(48)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(48, out_features)
        
    def forward(self, x):
        # x: (Batch, 1, Seq)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        fused = torch.cat([c3, c5, c7], dim=1) # (Batch, 48, Seq)
        fused = self.se(fused)
        pooled = self.pool(fused).squeeze(-1) # (Batch, 48)
        return self.proj(pooled) # (Batch, out_features)

class QuantumClassifierBackend(nn.Module):
    def __init__(self, num_qubits=8, num_layers=3, num_classes=3):
        super(QuantumClassifierBackend, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.weights = nn.Parameter(torch.randn(num_layers, num_qubits, 3))
        
        @qml.qnode(self.dev, interface="torch")
        def _qnode(inputs, weights):
            # Data Re-uploading architecture
            for i in range(self.num_layers):
                qml.AngleEmbedding(inputs, wires=range(self.num_qubits))
                qml.StronglyEntanglingLayers(weights[i:i+1], wires=range(self.num_qubits))
                
            # Expectation values for classification + Von Neumann Entropy for regularization
            exp_vals = [qml.expval(qml.PauliZ(w)) for w in range(self.num_classes)]
            entropy = qml.vn_entropy(wires=range(self.num_qubits // 2))
            return tuple(exp_vals) + (entropy,)
            
        self.qnode = _qnode
        
    def forward(self, x):
        # x: (Batch, num_qubits)
        x = torch.tanh(x) * np.pi # scale to [-pi, pi]
        
        batch_size = x.shape[0]
        logits = []
        entropies = []
        
        for i in range(batch_size):
            res = self.qnode(x[i], self.weights)
            # res is a tuple of (num_classes + 1) tensors
            res_t = torch.stack(res)
            logits.append(res_t[:self.num_classes])
            entropies.append(res_t[-1])
            
        logits = torch.stack(logits) # (Batch, num_classes)
        entropies = torch.stack(entropies) # (Batch,)
        
        return logits, entropies

class MultiScaleQuantumModel(nn.Module):
    def __init__(self, num_features=5, num_classes=3):
        super(MultiScaleQuantumModel, self).__init__()
        self.aug = NoiseAwareAugmentation()
        # Create a block for each modality
        self.blocks = nn.ModuleList([
            MultiScaleModalityBlock(in_channels=1, out_features=8)
            for _ in range(num_features)
        ])
        # Project concatenated modalities to 8 features
        self.proj = nn.Sequential(
            nn.Linear(num_features * 8, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8)
        )
        self.quantum_backend = QuantumClassifierBackend(num_qubits=8, num_layers=3, num_classes=num_classes)
        
    def forward(self, x):
        # x: (Batch, Seq, Features)
        x = self.aug(x)
        # Permute for CNN: (Batch, Features, Seq)
        x_perm = x.permute(0, 2, 1)
        
        modality_features = []
        for i, block in enumerate(self.blocks):
            f_i = block(x_perm[:, i:i+1, :]) # (Batch, 8)
            modality_features.append(f_i)
            
        fused = torch.cat(modality_features, dim=1) # (Batch, 40)
        proj_features = self.proj(fused) # (Batch, 8)
        
        return self.quantum_backend(proj_features)

class LSTMQuantumModel(nn.Module):
    def __init__(self, num_features=5, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(LSTMQuantumModel, self).__init__()
        self.aug = NoiseAwareAugmentation()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 8)
        )
        self.quantum_backend = QuantumClassifierBackend(num_qubits=8, num_layers=3, num_classes=num_classes)
        
    def forward(self, x):
        x = self.aug(x)
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1) # (Batch, Hidden*2)
        proj_features = self.proj(pooled) # (Batch, 8)
        return self.quantum_backend(proj_features)

class CNNLSTMQuantumModel(nn.Module):
    def __init__(self, num_features=5, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(CNNLSTMQuantumModel, self).__init__()
        self.aug = NoiseAwareAugmentation()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 8)
        )
        self.quantum_backend = QuantumClassifierBackend(num_qubits=8, num_layers=3, num_classes=num_classes)
        
    def forward(self, x):
        x = self.aug(x)
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn(x_cnn)
        x_lstm_in = x_cnn.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm_in)
        last_out = lstm_out[:, -1, :] # (Batch, Hidden*2)
        proj_features = self.proj(last_out)
        return self.quantum_backend(proj_features)

class TransformerQuantumModel(nn.Module):
    def __init__(self, num_features=5, d_model=64, nhead=8, num_layers=2, dim_feedforward=128, num_classes=3, dropout=0.1, attn_type='flash'):
        super(TransformerQuantumModel, self).__init__()
        self.aug = NoiseAwareAugmentation()
        self.embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.attn_type = attn_type
        if attn_type == 'linear':
            self.layers = nn.ModuleList([FastLinearAttentionLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
            self.transformer_encoder = None
        elif attn_type == 'flash':
            self.layers = nn.ModuleList([FastTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
            self.transformer_encoder = None
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.layers = None
            
        self.proj = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 8)
        )
        self.quantum_backend = QuantumClassifierBackend(num_qubits=8, num_layers=3, num_classes=num_classes)
        
    def forward(self, x):
        x = self.aug(x)
        x_emb = self.embedding(x)
        x_pos = self.pos_encoder(x_emb)
        
        if self.layers is not None:
            for layer in self.layers:
                x_pos = layer(x_pos)
            x_enc = x_pos
        else:
            x_enc = self.transformer_encoder(x_pos)
            
        pooled = x_enc.mean(dim=1)
        proj_features = self.proj(pooled)
        return self.quantum_backend(proj_features)


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import numpy as np
    # Smoke Test
    batch_size = 4
    seq_len = 3000 # 30s at 100Hz
    features = 5
    
    dummy_input = torch.randn(batch_size, seq_len, features)
    
    print("Testing LSTM Model...")
    lstm = LSTMModel()
    out_lstm = lstm(dummy_input)
    print("LSTM Output Shape:", out_lstm.shape)
    
    print("\nTesting CNN-LSTM Model...")
    cnnlstm = CNNLSTMModel()
    out_cnnlstm = cnnlstm(dummy_input)
    print("CNN-LSTM Output Shape:", out_cnnlstm.shape)

    print("\nTesting Transformer Model...")
    transformer = TransformerModel()
    out_transformer = transformer(dummy_input)
    print("Transformer Output Shape:", out_transformer.shape)

    print("\nTesting Classical Baseline Model...")
    baseline = ClassicalBaseline()
    out_baseline = baseline(dummy_input)
    print("Baseline Output Shape:", out_baseline.shape)

    print("\nTesting Multi-Scale CNN Model...")
    mscnn = MultiScaleCNNModel()
    out_mscnn = mscnn(dummy_input)
    print("Multi-Scale CNN Output Shape:", out_mscnn.shape)

    print("\nTesting Hybrid Quantum-Classical Models...")
    try:
        models = [
            ("MultiScale Quantum", MultiScaleQuantumModel()),
            ("LSTM Quantum", LSTMQuantumModel()),
            ("CNN-LSTM Quantum", CNNLSTMQuantumModel()),
            ("Transformer Quantum", TransformerQuantumModel())
        ]
        dummy_targets = torch.randint(0, 3, (batch_size,))
        criterion = HybridLoss()
        
        for name, model in models:
            logits, entropies = model(dummy_input)
            loss = criterion(logits, entropies, dummy_targets)
            print(f"{name} Output Shape: {logits.shape}, Entropy Avg: {entropies.mean().item():.4f}, Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"Error testing quantum models: {e}")
