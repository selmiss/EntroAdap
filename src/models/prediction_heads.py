"""Regression head module for molecular property prediction tasks."""

import torch
import torch.nn as nn
from typing import Optional


class RegressionHead(nn.Module):
    """
    Regression head for predicting continuous values from LLM hidden states.
    
    This head extracts features from the last hidden state and projects to a single
    continuous value, which is much more efficient and reliable than generating
    numbers token-by-token.
    """
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        pooling_strategy: str = "last",
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize regression head.
        
        Args:
            hidden_size: Size of LLM hidden states
            dropout: Dropout probability
            pooling_strategy: How to pool sequence representations
                - "last": Use last token's hidden state
                - "mean": Average all tokens
                - "attention": Learned attention pooling
            hidden_dim: Optional intermediate hidden dimension for projection
        """
        super().__init__()
        
        self.pooling_strategy = pooling_strategy
        
        # Pooling mechanism
        if pooling_strategy == "attention":
            self.attention = nn.Linear(hidden_size, 1)
        
        # Projection layers
        if hidden_dim is not None:
            # Two-layer projection with non-linearity
            self.projector = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # Single layer projection
            self.projector = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
    
    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence hidden states to a single vector per sample.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] mask of valid tokens
        
        Returns:
            pooled: [batch_size, hidden_size]
        """
        if self.pooling_strategy == "last":
            # Use last non-padding token
            if attention_mask is not None:
                # Find last valid position for each sequence
                seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths]
            else:
                # No mask, just use last token
                pooled = hidden_states[:, -1, :]
        
        elif self.pooling_strategy == "mean":
            # Average over valid tokens
            if attention_mask is not None:
                # Mask out padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                # No mask, just average all
                pooled = hidden_states.mean(dim=1)
        
        elif self.pooling_strategy == "attention":
            # Learned attention pooling
            attention_scores = self.attention(hidden_states).squeeze(-1)  # [batch, seq_len]
            
            if attention_mask is not None:
                # Mask padding tokens
                attention_scores = attention_scores.masked_fill(
                    attention_mask == 0, float('-inf')
                )
            
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
            pooled = (hidden_states * attention_weights).sum(dim=1)  # [batch, hidden_size]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of regression head.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] from LLM
            attention_mask: [batch_size, seq_len] mask of valid tokens
        
        Returns:
            predictions: [batch_size, 1] continuous predictions
        """
        # Pool to single vector per sample
        pooled = self.pool_hidden_states(hidden_states, attention_mask)  # [batch, hidden_size]
        
        # Project to scalar
        predictions = self.projector(pooled)  # [batch, 1]
        
        return predictions


class ClassificationHead(nn.Module):
    """
    Classification head for categorical prediction tasks.
    
    Useful for tasks like:
    - Binary classification (active/inactive)
    - Multi-class classification (drug categories)
    - Multi-label classification (multiple properties)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        pooling_strategy: str = "last",
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize classification head.
        
        Args:
            hidden_size: Size of LLM hidden states
            num_labels: Number of output classes
            dropout: Dropout probability
            pooling_strategy: How to pool sequence representations
            hidden_dim: Optional intermediate hidden dimension
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy
        
        # Reuse pooling from RegressionHead
        self.regression_head = RegressionHead(
            hidden_size=hidden_size,
            dropout=dropout,
            pooling_strategy=pooling_strategy,
            hidden_dim=None,  # We'll build our own projector
        )
        
        # Classification projector
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of classification head.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] from LLM
            attention_mask: [batch_size, seq_len] mask of valid tokens
        
        Returns:
            logits: [batch_size, num_labels] class logits
        """
        # Pool to single vector per sample
        pooled = self.regression_head.pool_hidden_states(hidden_states, attention_mask)
        
        # Project to class logits
        logits = self.classifier(pooled)  # [batch, num_labels]
        
        return logits
