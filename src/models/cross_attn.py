import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 4
        
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class FusionBlock(nn.Module):
    """Fusion block with self-attention, cross-attention, and feed-forward layers."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the fusion block."""
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states, _ = self.cross_attn(
            query=hidden_states,
            key=key_value_states,
            value=key_value_states,
            key_padding_mask=cross_key_padding_mask,
            need_weights=False,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def freeze(self):
        """Freeze all parameters in the fusion block."""
        for param in self.parameters():
            param.requires_grad = False