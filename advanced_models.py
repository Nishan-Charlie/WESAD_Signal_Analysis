import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml

# --- Building Blocks ---

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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

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

# --- Backbones ---

class LSTMBackbone(nn.Module):
    def __init__(self, in_channels, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMBackbone, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out.mean(dim=1) # (Batch, 2*hidden_size)

class CNNBackbone(nn.Module):
    def __init__(self, in_channels, out_channels=16):
        super(CNNBackbone, self).__init__()
        self.ms_cnn = MultiScale1DCNN(in_channels, out_channels)
        self.se = SEBlock1D(out_channels * 3)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        # x: (Batch, Seq, Feats) -> (Batch, Feats, Seq)
        x = x.permute(0, 2, 1)
        x = self.ms_cnn(x)
        x = self.se(x)
        return self.pool(x).squeeze(-1) # (Batch, 3*out_channels)

class TransformerBackbone(nn.Module):
    def __init__(self, in_channels, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super(TransformerBackbone, self).__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([FastTransformerEncoderLayer(d_model, nhead, 256, dropout) for _ in range(num_layers)])
    def forward(self, x):
        x = self.proj(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=1) # (Batch, d_model)

# --- Backends ---

class QuantumClassifierBackend(nn.Module):
    def __init__(self, num_qubits=10, num_layers=4, num_classes=3):
        super(QuantumClassifierBackend, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dev = qml.device("lightning.qubit", wires=self.num_qubits)
        self.weights = nn.Parameter(torch.randn(num_layers, num_qubits, 3))
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def _qnode(inputs, weights):
            for i in range(self.num_layers):
                qml.AngleEmbedding(inputs, wires=range(self.num_qubits))
                qml.StronglyEntanglingLayers(weights[i:i+1], wires=range(self.num_qubits))
            return tuple(qml.expval(qml.PauliZ(w)) for w in range(self.num_classes))
        self.qnode = _qnode
    def forward(self, x):
        x = torch.tanh(x) * np.pi
        res = self.qnode(x, self.weights)
        logits = torch.stack(res, dim=-1) if isinstance(res, tuple) else res
        return logits, torch.zeros(x.shape[0], device=x.device)

# --- Universal Model ---

class UniversalMultimodalModel(nn.Module):
    def __init__(self, backbone_type='cnn', fusion_type='mid', is_quantum=False, num_features=5, num_classes=3):
        super(UniversalMultimodalModel, self).__init__()
        self.backbone_type = backbone_type
        self.fusion_type = fusion_type
        self.is_quantum = is_quantum
        self.num_features = num_features

        def make_backbone(in_ch):
            if backbone_type == 'lstm': return LSTMBackbone(in_ch)
            if backbone_type == 'cnn': return CNNBackbone(in_ch)
            if backbone_type == 'transformer': return TransformerBackbone(in_ch)
            return CNNBackbone(in_ch)

        if fusion_type == 'early':
            self.backbone = make_backbone(num_features)
            feat_dim = self._get_dim(num_features)
        elif fusion_type in ['mid', 'late']:
            self.branches = nn.ModuleList([make_backbone(1) for _ in range(num_features)])
            feat_dim = self._get_dim(1) * num_features
        
        if fusion_type == 'late':
            branch_dim = self._get_dim(1)
            self.branch_heads = nn.ModuleList([nn.Linear(branch_dim, num_classes) for _ in range(num_features)])
        else:
            if is_quantum:
                self.proj = nn.Linear(feat_dim, 10)
                self.backend = QuantumClassifierBackend(10, 4, num_classes)
            else:
                self.backend = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def _get_dim(self, in_ch):
        if self.backbone_type == 'lstm': return 128
        if self.backbone_type == 'cnn': return 48
        if self.backbone_type == 'transformer': return 64
        return 48

    def forward(self, x):
        if self.fusion_type == 'early':
            f = self.backbone(x)
            out = self.backend(f if not self.is_quantum else self.proj(f))
        elif self.fusion_type == 'mid':
            b_f = [self.branches[i](x[:, :, i:i+1]) for i in range(self.num_features)]
            f = torch.cat(b_f, dim=-1)
            out = self.backend(f if not self.is_quantum else self.proj(f))
        elif self.fusion_type == 'late':
            logits = [self.branch_heads[i](self.branches[i](x[:, :, i:i+1])) for i in range(self.num_features)]
            out = torch.stack(logits).mean(dim=0)
        return out

# --- Utilities ---

class NoiseAwareAugmentation(nn.Module):
    def __init__(self, noise_std=0.01, bw_amplitude=0.05):
        super(NoiseAwareAugmentation, self).__init__()
        self.noise_std, self.bw_amplitude = noise_std, bw_amplitude
    def forward(self, x):
        if not self.training or x.dim() != 3: return x
        b, s, f = x.shape
        noise = torch.randn_like(x) * self.noise_std
        t = torch.arange(s, dtype=x.dtype, device=x.device) / s
        cyc = torch.empty(b, 1, f, device=x.device).uniform_(1, 5)
        ph = torch.empty(b, 1, f, device=x.device).uniform_(0, 2*np.pi)
        bw = self.bw_amplitude * torch.sin(2 * np.pi * cyc * t.view(1, s, 1) + ph)
        return x + noise + bw

class HybridLoss(nn.Module):
    def __init__(self, lambda_entropy=0.1):
        super(HybridLoss, self).__init__()
        self.lambda_entropy, self.ce = lambda_entropy, nn.CrossEntropyLoss()
    def forward(self, logits, entropies, targets):
        return self.ce(logits, targets) + self.lambda_entropy * entropies.mean()

# --- Legacy Wrappers ---
class LSTMModel(UniversalMultimodalModel):
    def __init__(self, **kwargs): super().__init__(backbone_type='lstm', fusion_type='early', **kwargs)
class CNNLSTMModel(UniversalMultimodalModel):
    def __init__(self, **kwargs): super().__init__(backbone_type='cnnlstm', fusion_type='early', **kwargs)
class TransformerModel(UniversalMultimodalModel):
    def __init__(self, **kwargs): super().__init__(backbone_type='transformer', fusion_type='early', **kwargs)
class ClassicalBaseline(UniversalMultimodalModel):
    def __init__(self, **kwargs): super().__init__(backbone_type='cnn', fusion_type='mid', **kwargs)
class MultiScaleCNNModel(UniversalMultimodalModel):
    def __init__(self, **kwargs): super().__init__(backbone_type='cnn', fusion_type='early', **kwargs)
class MultiScaleQuantumModel(UniversalMultimodalModel):
    def __init__(self, **kwargs): super().__init__(backbone_type='cnn', fusion_type='mid', is_quantum=True, **kwargs)
class LSTMQuantumModel(UniversalMultimodalModel):
    def __init__(self, **kwargs): super().__init__(backbone_type='lstm', fusion_type='early', is_quantum=True, **kwargs)
class CNNLSTMQuantumModel(UniversalMultimodalModel):
    def __init__(self, **kwargs): super().__init__(backbone_type='cnnlstm', fusion_type='early', is_quantum=True, **kwargs)
class TransformerQuantumModel(UniversalMultimodalModel):
    def __init__(self, **kwargs): super().__init__(backbone_type='transformer', fusion_type='early', is_quantum=True, **kwargs)
class FlexibleClassicalModel(UniversalMultimodalModel):
    def __init__(self, fusion_type='mid', **kwargs): super().__init__(backbone_type='cnn', fusion_type=fusion_type, **kwargs)
