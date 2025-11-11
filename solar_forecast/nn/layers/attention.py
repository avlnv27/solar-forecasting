import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, 1))   # poids d'attention
        self.b = nn.Parameter(torch.zeros(1))              # biais d'attention

    def forward(self, x):
        """
        x: Tensor (batch, seq_len, input_dim)
        """
        # Compute attention scores (“how relevant is this timestep?”)
        # e = tanh(x @ W + b)
        e = torch.tanh(torch.matmul(x, self.W) + self.b)   # (B, seq_len, 1)
        e = e.squeeze(-1)                                 # (B, seq_len)

        # Normalized importante weights (all sum to 1)
        alpha = F.softmax(e, dim=1).unsqueeze(-1)          # (B, seq_len, 1)

        # Context vector as weighted sum of x
        context = torch.sum(x * alpha, dim=1)              # (B, input_dim)

        return context

