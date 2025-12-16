#!/usr/bin/env python3
# coding=utf-8
"""
Tests for SFT trainer configuration and environment variable expansion.
"""

import pytest
import os
from dataclasses import dataclass, field
from typing import Optional, Dict

from src.configs import ScriptArguments, SFTConfig
from src.trainer.sft import expand_env_vars


class TestSFTConfig:
    """Tests for SFT configuration handling."""
    
    def test_expand_env_vars_output_dir(self):
        """Test environment variable expansion in output_dir."""
        # Set environment variable
        os.environ["CHECKPOINT_DIR"] = "/tmp/checkpoints"
        
        script_args = ScriptArguments(dataset_name="test_data")
        training_args = SFTConfig(
            output_dir="${CHECKPOINT_DIR}/test_model",
            per_device_train_batch_size=1,
        )
        
        expand_env_vars(script_args, training_args)
        
        assert training_args.output_dir == "/tmp/checkpoints/test_model"
        
        # Clean up
        del os.environ["CHECKPOINT_DIR"]
    
    def test_expand_env_vars_dataset_name(self):
        """Test environment variable expansion in dataset_name."""
        os.environ["DATA_DIR"] = "/tmp/data"
        
        script_args = ScriptArguments(dataset_name="${DATA_DIR}/train.jsonl")
        training_args = SFTConfig(
            output_dir="./output",
            per_device_train_batch_size=1,
        )
        
        expand_env_vars(script_args, training_args)
        
        assert script_args.dataset_name == "/tmp/data/train.jsonl"
        
        # Clean up
        del os.environ["DATA_DIR"]
    
    def test_expand_env_vars_dataset_mixture_attribute(self):
        """Test that expand_env_vars uses dataset_mixture, not dataset_mixer."""
        # This test ensures we don't regress on the AttributeError bug
        script_args = ScriptArguments(dataset_name="test_data")
        training_args = SFTConfig(
            output_dir="./output",
            per_device_train_batch_size=1,
        )
        
        # Should not raise AttributeError about dataset_mixer
        try:
            expand_env_vars(script_args, training_args)
        except AttributeError as e:
            if "dataset_mixer" in str(e):
                pytest.fail(f"expand_env_vars incorrectly uses 'dataset_mixer' instead of 'dataset_mixture': {e}")
            else:
                raise
    
    def test_script_arguments_has_dataset_mixture(self):
        """Test that ScriptArguments has dataset_mixture attribute."""
        script_args = ScriptArguments(dataset_name="test")
        
        # Should have dataset_mixture attribute, not dataset_mixer
        assert hasattr(script_args, 'dataset_mixture')
        assert script_args.dataset_mixture is None  # Default value
        
        # Should NOT have dataset_mixer
        assert not hasattr(script_args, 'dataset_mixer')
    
    def test_expand_env_vars_with_none_values(self):
        """Test expand_env_vars handles None values gracefully."""
        # ScriptArguments requires at least dataset_name to be set
        script_args = ScriptArguments(dataset_name="test_data")
        training_args = SFTConfig(
            output_dir="./output",
            per_device_train_batch_size=1,
        )
        
        # Should not raise any errors
        expand_env_vars(script_args, training_args)
        
        assert script_args.dataset_name == "test_data"
    
    def test_expand_env_vars_no_env_vars(self):
        """Test expand_env_vars works when no environment variables are present."""
        script_args = ScriptArguments(dataset_name="data/train.jsonl")
        training_args = SFTConfig(
            output_dir="./output",
            per_device_train_batch_size=1,
        )
        
        expand_env_vars(script_args, training_args)
        
        # Values should remain unchanged
        assert script_args.dataset_name == "data/train.jsonl"
        assert training_args.output_dir == "./output"


class TestDatasetMixtureConfig:
    """Tests for dataset mixture configuration."""
    
    def test_dataset_mixture_vs_dataset_name(self):
        """Test that either dataset_name or dataset_mixture can be set."""
        # Test with dataset_name
        script_args1 = ScriptArguments(dataset_name="test_data")
        assert script_args1.dataset_name == "test_data"
        assert script_args1.dataset_mixture is None
        
        # Test that at least one must be provided (validation check)
        with pytest.raises(ValueError, match="Either `dataset_name` or `dataset_mixture` must be provided"):
            script_args2 = ScriptArguments(dataset_name=None, dataset_mixture=None)
    
    def test_attribute_naming_consistency(self):
        """Ensure dataset_mixture is consistently named throughout the codebase."""
        from src.configs import ScriptArguments
        from utils.data import get_dataset
        
        # ScriptArguments should have dataset_mixture
        args = ScriptArguments(dataset_name="test")
        assert hasattr(args, 'dataset_mixture')
        
        # get_dataset function should check for dataset_mixture
        import inspect
        source = inspect.getsource(get_dataset)
        assert 'dataset_mixture' in source
        # Should not use dataset_mixer
        assert 'dataset_mixer' not in source or source.count('dataset_mixture') > source.count('dataset_mixer')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
