#!/usr/bin/env python3
# coding=utf-8
"""
Tests for model utility functions.
"""

import pytest
import torch
from transformers import AutoConfig
from trl import ModelConfig

from src.models.training_configs import SFTConfig, OctopusConfig
from utils.model_utils import get_custom_model, get_model_and_peft_config
from src.models import Octopus


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
        return OctopusConfig(
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
        
        assert isinstance(model, Octopus)
        # Check that the model has the expected components
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'fusion_blocks')
        assert len(model.fusion_blocks) == 2
        # Check encoder hidden dim matches config
        assert model.encoder.hidden_dim == 256
    
    @pytest.mark.slow
    def test_get_custom_model_with_different_configs(
        self, model_args, training_args
    ):
        """Test loading custom model with different configurations."""
        # Test with larger model
        multimodal_args = OctopusConfig(
            use_custom_model=True,
            modality_vocab_size=5000,
            modality_embedding_dim=512,
            num_fusion_blocks=4,
            num_attention_heads=8,
            dropout=0.2,
        )
        
        model = get_custom_model(model_args, training_args, multimodal_args)
        
        assert isinstance(model, Octopus)
        # Check encoder hidden dim and fusion blocks
        assert model.encoder.hidden_dim == 512
        assert len(model.fusion_blocks) == 4
    
    @pytest.mark.slow
    def test_get_custom_model_with_custom_dims(
        self, model_args, training_args
    ):
        """Test loading custom model with custom hidden dimensions."""
        multimodal_args = OctopusConfig(
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
        
        assert isinstance(model, Octopus)
        # Check that custom dimensions are respected
        assert model.fusion_blocks[0].ffn.fc1.out_features == 2048

    @pytest.mark.unit
    def test_get_model_and_peft_config_branching(self, monkeypatch, model_args, training_args):
        """Unit test: branching/peft behavior without actually loading models."""
        sentinel_model = object()
        sentinel_peft = object()

        def fake_get_model(_model_args, _training_args):
            return sentinel_model

        def fake_get_peft_config(_model_args):
            return sentinel_peft

        def fake_get_custom_model(_model_args, _training_args, _mm_args):
            return sentinel_model

        import utils.model_utils as mu

        monkeypatch.setattr(mu, "get_model", fake_get_model)
        monkeypatch.setattr(mu, "get_peft_config", fake_get_peft_config)
        monkeypatch.setattr(mu, "get_custom_model", fake_get_custom_model)

        # Standard path => returns peft_config
        model, peft_config = get_model_and_peft_config(model_args, training_args, multimodal_args=None)
        assert model is sentinel_model
        assert peft_config is sentinel_peft

        # Multimodal path => peft_config must be None (handled inside get_custom_model)
        mm_args = OctopusConfig(use_custom_model=True)
        model, peft_config = get_model_and_peft_config(model_args, training_args, multimodal_args=mm_args)
        assert model is sentinel_model
        assert peft_config is None


class TestMultiModalConfig:
    """Tests for OctopusConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = OctopusConfig()
        
        assert config.use_custom_model is False
        assert config.modality_vocab_size == 10000
        assert config.modality_embedding_dim == 768
        assert config.num_fusion_blocks == 4
        assert config.num_attention_heads == 8
        assert config.dropout == 0.1
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = OctopusConfig(
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
