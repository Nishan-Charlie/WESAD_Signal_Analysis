import torch
import torch.nn as nn
import torch.nn.functional as F

class FastTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
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
        super().__init__()
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
        
        q_m = self._elu_map(q) # (B, H, S, D)
        k_m = self._elu_map(k)
        
        kv = torch.matmul(k_m.transpose(-2, -1), v) # (B, H, D, D)
        z = 1 / (torch.matmul(q_m, k_m.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
        
        attn_out = torch.matmul(q_m, kv) * z
        attn_out = attn_out.transpose(1, 2).reshape(B, S, E)
        attn_out = self.out_proj(attn_out)
        
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        ff_out = self.ff(src)
        src = src + self.dropout2(ff_out)
        return self.norm2(src)

B, S, E = 2, 256, 64
src = torch.randn(B, S, E)
flash = FastTransformerEncoderLayer(E, 8)
lin = FastLinearAttentionLayer(E, 8)

out1 = flash(src)
print("Flash shape:", out1.shape)
out2 = lin(src)
print("Linear shape:", out2.shape)
