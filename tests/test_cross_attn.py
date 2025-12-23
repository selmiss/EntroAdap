#!/usr/bin/env python3
# coding=utf-8
"""
Tests for fusion block components.
"""

import pytest
import torch
import torch.nn as nn

from src.models.cross_attn import FeedForward, FusionBlock


class TestFeedForward:
    """Tests for FeedForward network."""
    
    def test_initialization(self):
        """Test FeedForward initialization."""
        ffn = FeedForward(hidden_dim=768, intermediate_dim=3072, dropout=0.1)
        
        assert ffn.fc1.in_features == 768
        assert ffn.fc1.out_features == 3072
        assert ffn.fc2.in_features == 3072
        assert ffn.fc2.out_features == 768
    
    def test_default_intermediate_dim(self):
        """Test that intermediate_dim defaults to 4x hidden_dim."""
        ffn = FeedForward(hidden_dim=768, dropout=0.1)
        
        assert ffn.fc1.out_features == 768 * 4  # Default is 4x
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size, seq_len, hidden_dim = 2, 10, 768
        ffn = FeedForward(hidden_dim=hidden_dim, dropout=0.1)
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = ffn(x)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
    
    def test_forward_values(self):
        """Test that forward pass produces different values (not identity)."""
        batch_size, seq_len, hidden_dim = 2, 10, 768
        ffn = FeedForward(hidden_dim=hidden_dim, dropout=0.0)  # No dropout for determinism
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = ffn(x)
        
        # Output should be different from input
        assert not torch.allclose(output, x, atol=1e-3)


class TestFusionBlock:
    """Tests for FusionBlock."""
    
    def test_initialization(self):
        """Test FusionBlock initialization."""
        fusion_block = FusionBlock(
            hidden_dim=768,
            num_heads=8,
            intermediate_dim=3072,
            dropout=0.1,
        )
        
        assert isinstance(fusion_block.self_attn, nn.MultiheadAttention)
        assert isinstance(fusion_block.cross_attn, nn.MultiheadAttention)
        assert isinstance(fusion_block.ffn, FeedForward)
        assert isinstance(fusion_block.self_attn_norm, nn.LayerNorm)
        assert isinstance(fusion_block.cross_attn_norm, nn.LayerNorm)
        assert isinstance(fusion_block.ffn_norm, nn.LayerNorm)
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size, modality_seq_len, text_seq_len, hidden_dim = 2, 16, 32, 768
        fusion_block = FusionBlock(hidden_dim=hidden_dim, num_heads=8, dropout=0.1)
        
        modality_hidden = torch.randn(batch_size, modality_seq_len, hidden_dim)
        text_hidden = torch.randn(batch_size, text_seq_len, hidden_dim)
        
        output = fusion_block(
            hidden_states=modality_hidden,
            key_value_states=text_hidden,
        )
        
        assert output.shape == (batch_size, modality_seq_len, hidden_dim)
    
    def test_forward_with_padding_masks(self):
        """Test forward pass with padding masks."""
        batch_size, modality_seq_len, text_seq_len, hidden_dim = 2, 16, 32, 768
        fusion_block = FusionBlock(hidden_dim=hidden_dim, num_heads=8, dropout=0.1)
        
        modality_hidden = torch.randn(batch_size, modality_seq_len, hidden_dim)
        text_hidden = torch.randn(batch_size, text_seq_len, hidden_dim)
        
        # Create padding masks (True = padding, False = valid)
        key_padding_mask = torch.zeros(batch_size, modality_seq_len, dtype=torch.bool)
        key_padding_mask[:, -4:] = True  # Last 4 tokens are padding
        
        cross_key_padding_mask = torch.zeros(batch_size, text_seq_len, dtype=torch.bool)
        cross_key_padding_mask[:, -8:] = True  # Last 8 tokens are padding
        
        output = fusion_block(
            hidden_states=modality_hidden,
            key_value_states=text_hidden,
            key_padding_mask=key_padding_mask,
            cross_key_padding_mask=cross_key_padding_mask,
        )
        
        assert output.shape == (batch_size, modality_seq_len, hidden_dim)
    
    def test_residual_connections(self):
        """Test that residual connections are working."""
        batch_size, modality_seq_len, text_seq_len, hidden_dim = 2, 16, 32, 768
        fusion_block = FusionBlock(hidden_dim=hidden_dim, num_heads=8, dropout=0.0)
        
        # Set to eval mode to disable dropout
        fusion_block.eval()
        
        modality_hidden = torch.randn(batch_size, modality_seq_len, hidden_dim)
        text_hidden = torch.randn(batch_size, text_seq_len, hidden_dim)
        
        with torch.no_grad():
            output = fusion_block(
                hidden_states=modality_hidden,
                key_value_states=text_hidden,
            )
        
        # Output should be different from input (due to transformations)
        assert not torch.allclose(output, modality_hidden, atol=1e-3)
        
        # But output magnitude should be in reasonable range
        assert output.abs().mean() < 100.0
    
    def test_num_heads_divisibility(self):
        """Test that num_heads must divide hidden_dim evenly."""
        # This should work
        fusion_block = FusionBlock(hidden_dim=768, num_heads=8)
        assert fusion_block is not None
        
        # This should also work
        fusion_block = FusionBlock(hidden_dim=768, num_heads=12)
        assert fusion_block is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
