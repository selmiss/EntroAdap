#!/usr/bin/env python3
# coding=utf-8
"""
Tests for model utility functions.
"""

import pytest
import torch
from transformers import AutoConfig
from trl import ModelConfig

from src.configs import SFTConfig, MultiModalConfig
from utils.model_utils import get_custom_model
from src.models import MultiModalLLM


class TestModelUtils:
    """Tests for model utility functions."""
    
    @pytest.fixture
    def model_args(self):
        """Create model arguments for testing."""
        return ModelConfig(
            model_name_or_path="gpt2",
            dtype="float32",
            attn_implementation="eager",
        )
    
    @pytest.fixture
    def training_args(self):
        """Create training arguments for testing."""
        return SFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            gradient_checkpointing=False,
        )
    
    @pytest.fixture
    def multimodal_args(self):
        """Create multimodal arguments for testing."""
        return MultiModalConfig(
            use_custom_model=True,
            modality_vocab_size=1000,
            modality_embedding_dim=256,
            num_fusion_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )
    
    @pytest.mark.slow
    def test_get_custom_model(self, model_args, training_args, multimodal_args):
        """Test loading custom multimodal model."""
        model = get_custom_model(model_args, training_args, multimodal_args)
        
        assert isinstance(model, MultiModalLLM)
        assert model.modality_embedding.num_embeddings == 1000
        assert model.modality_embedding.embedding_dim == 256
        assert len(model.fusion_blocks) == 2
    
    @pytest.mark.slow
    def test_get_custom_model_with_different_configs(
        self, model_args, training_args
    ):
        """Test loading custom model with different configurations."""
        # Test with larger model
        multimodal_args = MultiModalConfig(
            use_custom_model=True,
            modality_vocab_size=5000,
            modality_embedding_dim=512,
            num_fusion_blocks=4,
            num_attention_heads=8,
            dropout=0.2,
        )
        
        model = get_custom_model(model_args, training_args, multimodal_args)
        
        assert isinstance(model, MultiModalLLM)
        assert model.modality_embedding.num_embeddings == 5000
        assert model.modality_embedding.embedding_dim == 512
        assert len(model.fusion_blocks) == 4
    
    @pytest.mark.slow
    def test_get_custom_model_with_custom_dims(
        self, model_args, training_args
    ):
        """Test loading custom model with custom hidden dimensions."""
        multimodal_args = MultiModalConfig(
            use_custom_model=True,
            modality_vocab_size=1000,
            modality_embedding_dim=256,
            num_fusion_blocks=2,
            num_attention_heads=4,
            fusion_hidden_dim=512,
            fusion_intermediate_dim=2048,
            dropout=0.1,
        )
        
        model = get_custom_model(model_args, training_args, multimodal_args)
        
        assert isinstance(model, MultiModalLLM)
        # Check that custom dimensions are respected
        assert model.fusion_blocks[0].ffn.fc1.out_features == 2048


class TestMultiModalConfig:
    """Tests for MultiModalConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MultiModalConfig()
        
        assert config.use_custom_model is False
        assert config.modality_vocab_size == 10000
        assert config.modality_embedding_dim == 768
        assert config.num_fusion_blocks == 4
        assert config.num_attention_heads == 8
        assert config.dropout == 0.1
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = MultiModalConfig(
            use_custom_model=True,
            modality_vocab_size=5000,
            modality_embedding_dim=512,
            num_fusion_blocks=6,
            num_attention_heads=16,
            dropout=0.2,
        )
        
        assert config.use_custom_model is True
        assert config.modality_vocab_size == 5000
        assert config.modality_embedding_dim == 512
        assert config.num_fusion_blocks == 6
        assert config.num_attention_heads == 16
        assert config.dropout == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
