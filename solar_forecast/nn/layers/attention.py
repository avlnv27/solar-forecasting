import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Additive attention mechanism (Bahdanau-style).
    
    FIXED: Uses nn.Linear for proper weight initialization
    instead of manual nn.Parameter with torch.randn.
    
    This ensures proper Xavier/Kaiming initialization and
    better training stability.
    """

    def __init__(self, input_dim):
        super().__init__()
        # FIXED: Use nn.Linear instead of manual parameters
        # This gives proper initialization (Xavier uniform by default)
        self.W = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        """
        Compute attention-weighted sum over sequence dimension.
        
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            context: Tensor of shape (batch, input_dim)
                    - weighted sum of x over seq_len dimension
        """
        # Compute attention scores
        e = torch.tanh(self.W(x))  # (B, seq_len, 1)
        e = e.squeeze(-1)           # (B, seq_len)

        # Normalize to get attention weights (sum to 1)
        alpha = F.softmax(e, dim=1).unsqueeze(-1)  # (B, seq_len, 1)

        # Compute context as weighted sum
        context = torch.sum(x * alpha, dim=1)  # (B, input_dim)

        return context


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Optional enhancement if you want to try more sophisticated attention.
    """
    
    def __init__(self, input_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert input_dim % n_heads == 0, "input_dim must be divisible by n_heads"
        
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            out: (batch, input_dim) - attention-pooled output
        """
        B, L, D = x.shape
        
        # Linear projections and split into heads
        Q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, D)
        
        # Output projection
        out = self.W_o(attn_output)
        
        # Pool over sequence dimension (mean)
        out = out.mean(dim=1)  # (B, D)
        
        return out


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention (Transformer-style).
    
    Another alternative if you want to experiment.
    """
    
    def __init__(self, input_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim or input_dim
        
        self.q_proj = nn.Linear(input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(input_dim, self.hidden_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_dim]))
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            context: (batch, input_dim)
        """
        Q = self.q_proj(x)  # (B, L, hidden_dim)
        K = self.k_proj(x)  # (B, L, hidden_dim)
        V = self.v_proj(x)  # (B, L, input_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        
        # Pool over sequence (weighted sum using mean of attention weights)
        context = context.mean(dim=1)  # Simple mean pooling
        
        return context