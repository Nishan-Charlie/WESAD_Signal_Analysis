import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScale1DCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        """
        Input Backbone: Multi-Scale 1D-CNN using 3 parallel paths.
        Kernel sizes: 3, 7, and 11 to capture different temporal resolutions.
        """
        super(MultiScale1DCNN, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv11 = nn.Conv1d(in_channels, out_channels, kernel_size=11, padding=5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        c3 = self.relu(self.conv3(x))
        c7 = self.relu(self.conv7(x))
        c11 = self.relu(self.conv11(x))
        # Concatenate features from all temporal scales along channel dimension
        return torch.cat([c3, c7, c11], dim=1)


class TemporalAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        """
        Attention Mechanism: 1D Self-Attention block to weigh the importance 
        of different time-segments (e.g. focusing on R-peaks in ECG).
        """
        super(TemporalAttention1D, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        x_permuted = x.permute(0, 2, 1) # Reshape to (Batch, Sequence_Length, Channels) required for PyTorch MHA
        
        # Self attention over time sequence
        attn_out, _ = self.mha(x_permuted, x_permuted, x_permuted)
        
        # Residual connection + LayerNorm
        out = self.layer_norm(x_permuted + attn_out)
        
        # Revert layout: (Batch, Channels, Sequence_Length)
        return out.permute(0, 2, 1)


class ModalityBranch(nn.Module):
    def __init__(self, in_channels=1, cnn_out_channels=16, num_heads=4):
        super(ModalityBranch, self).__init__()
        # 1. Multi-scale Convolutional Feature Extraction
        self.ms_cnn = MultiScale1DCNN(in_channels, cnn_out_channels)
        
        # Output channels from MS-CNN is cnn_out_channels * 3 (due to concatenation)
        embed_dim = cnn_out_channels * 3
        
        # Optional Max Pooling to reduce sequence length and memory consumption footprint
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 2. Temporal Self-Attention
        self.attention = TemporalAttention1D(embed_dim=embed_dim, num_heads=num_heads)
        
    def forward(self, x):
        x = self.ms_cnn(x)
        x = self.pool(x)
        x = self.attention(x)
        return x


class ClassicalBaseline(nn.Module):
    def __init__(self, in_channels=1, latent_dim=8, num_classes=3):
        super(ClassicalBaseline, self).__init__()
        
        # Base filter size
        cnn_out_channels = 16
        
        # Modality specific branches
        self.branch_ecg = ModalityBranch(in_channels, cnn_out_channels)
        self.branch_eda = ModalityBranch(in_channels, cnn_out_channels)
        self.branch_emg = ModalityBranch(in_channels, cnn_out_channels)
        self.branch_resp = ModalityBranch(in_channels, cnn_out_channels)
        
        # Total channels after concatenating 4 modalities: 4 * (16 * 3) = 192 features
        fused_channels = 4 * 3 * cnn_out_channels
        
        # Dimensionality Reduction Step
        # 1x1 Conv + Global Average Pooling -> Latent vector of size 8
        self.dim_reduction = nn.Sequential(
            nn.Conv1d(in_channels=fused_channels, out_channels=latent_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Replaces entire time dimension with 1
        )
        
        # Classification Head (3-layer MLP pivot for Classical Baseline)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_ecg, x_eda, x_emg, x_resp):
        # 1. Process each physiological modality individually
        f_ecg = self.branch_ecg(x_ecg)
        f_eda = self.branch_eda(x_eda)
        f_emg = self.branch_emg(x_emg)
        f_resp = self.branch_resp(x_resp)
        
        # 2. Multimodal Fusion (Concatenation along channel dimension)
        # Shape: (Batch, 192, Sequence_Length/2)
        fused = torch.cat([f_ecg, f_eda, f_emg, f_resp], dim=1)
        
        # 3. Dimensionality Reduction to latent vector of size 8
        # Shape becomes (Batch, 8, 1) after 1x1 conv + GAP
        latent = self.dim_reduction(fused)
        latent = latent.squeeze(-1) # Shape: (Batch, 8) mapping directly to target
        
        # 4. Classification via Fully Connected MLP
        out = self.classifier(latent)
        return out


if __name__ == "__main__":
    # Smoke Test script execution
    print("Initializing Classical Baseline Model...")
    model = ClassicalBaseline()
    
    # Dummy data equivalent to batch generated in train.py 
    # (Batch Size 2, 1 Channel, Sequence Length 100)
    x_ecg = torch.randn(2, 1, 100)
    x_eda = torch.randn(2, 1, 100)
    x_emg = torch.randn(2, 1, 100)
    x_resp = torch.randn(2, 1, 100)
    
    # Forward Pass
    output = model(x_ecg, x_eda, x_emg, x_resp)
    
    print("\n--- Model Evaluation details ---")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters())}")
    print("Output Shape (Batch, Classes):", output.shape)
    print("Output Probabilities [No Stress, Low Stress, High Stress]:")
    print(output.detach())
    print("Feature dimensions successfully reduced to map exact Quantum expectations (n_qubits=8 vector space)!")
