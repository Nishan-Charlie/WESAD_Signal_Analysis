import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, num_features=5, d_model=64, nhead=8, num_layers=3, dim_feedforward=256, num_classes=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # 1. Input Projection
        self.embedding = nn.Linear(num_features, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        x = self.transformer_encoder(x) # (Batch, Seq, d_model)
        
        # Global Average Pooling across the sequence dimension
        x = x.mean(dim=1) # (Batch, d_model)
        
        logits = self.classifier(x)
        return logits


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
