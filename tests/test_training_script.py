"""
Tests for Training Script
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch

from src.runner.aa_encoder import (
    ModelArguments,
    DataArguments,
    ReconstructionTrainerWrapper,
)
from src.models.aa_encoder import AAEncoder
from src.trainer.reconstruction import ReconstructionTrainer


class TestModelArguments:
    """Test model arguments dataclass."""
    
    def test_default_values(self):
        """Test default argument values."""
        args = ModelArguments()
        assert args.hidden_dim == 256
        assert args.num_layers == 6
        assert args.dropout == 0.1
        assert args.update_coords is True
        assert args.use_layernorm is True
        assert args.num_rbf == 32
        assert args.rbf_max == 10.0
        assert args.num_elements == 119
        assert args.num_dist_bins == 64
    
    def test_custom_values(self):
        """Test custom argument values."""
        args = ModelArguments(
            hidden_dim=128,
            num_layers=4,
            dropout=0.2,
        )
        assert args.hidden_dim == 128
        assert args.num_layers == 4
        assert args.dropout == 0.2


class TestDataArguments:
    """Test data arguments dataclass."""
    
    def test_required_fields(self):
        """Test required fields."""
        args = DataArguments(train_data_path="/path/to/train.parquet")
        assert args.train_data_path == "/path/to/train.parquet"
        assert args.val_data_path is None
    
    def test_optional_fields(self):
        """Test optional fields."""
        args = DataArguments(
            train_data_path="/path/to/train.parquet",
            val_data_path="/path/to/val.parquet",
            node_mask_prob=0.2,
        )
        assert args.val_data_path == "/path/to/val.parquet"
        assert args.node_mask_prob == 0.2


class TestTrainerWrapper:
    """Test custom trainer wrapper."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        encoder = AAEncoder(hidden_dim=64, num_layers=2)
        return ReconstructionTrainer(encoder=encoder)
    
    @pytest.fixture
    def mock_batch(self):
        """Create mock batch."""
        N = 20
        E = 40
        
        # Create protein-like data
        node_feat = torch.cat([
            torch.randint(0, 119, (N, 1)),
            torch.randint(0, 46, (N, 1)),
            torch.randint(0, 24, (N, 1)),
            torch.randint(0, 27, (N, 1)),
            torch.randint(1, 100, (N, 1)),
            torch.randint(0, 2, (N, 1)),
            torch.randint(0, 2, (N, 1)),
        ], dim=1).float()
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_attr': torch.rand(E, 1),
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': torch.randn(N, 3),
                'batch': torch.zeros(N, dtype=torch.long),
            }
        }
        
        return {
            'data': data,
            'batch': torch.zeros(N, dtype=torch.long),
            'node_mask': torch.rand(N) < 0.15,
            'edge_mask': torch.rand(E) < 0.15,
            'element_labels': torch.randint(0, 119, (N,)),
            'dist_labels': torch.rand(E) * 15.0,
            'noise_labels': torch.randn(N, 3) * 0.1,
        }
    
    def test_move_to_device(self, model):
        """Test moving nested data structures to device."""
        from transformers import TrainingArguments
        
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=2,
            )
            
            trainer = ReconstructionTrainerWrapper(
                model=model,
                args=training_args,
            )
            
            # Test nested dict
            data = {
                'modality': 'protein',
                'value': {
                    'node_feat': torch.randn(10, 7),
                    'pos': torch.randn(10, 3),
                },
            }
            
            device = torch.device('cpu')
            moved = trainer._move_to_device(data, device)
            
            assert moved['modality'] == 'protein'
            assert moved['value']['node_feat'].device == device
            assert moved['value']['pos'].device == device
    
    def test_compute_loss(self, model, mock_batch):
        """Test loss computation."""
        from transformers import TrainingArguments
        
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=2,
                logging_steps=1,
            )
            
            trainer = ReconstructionTrainerWrapper(
                model=model,
                args=training_args,
            )
            
            loss = trainer.compute_loss(model, mock_batch, return_outputs=False)
            
            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0  # Scalar
            assert loss.item() >= 0
    
    def test_compute_loss_with_outputs(self, model, mock_batch):
        """Test loss computation with outputs."""
        from transformers import TrainingArguments
        
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=2,
            )
            
            trainer = ReconstructionTrainerWrapper(
                model=model,
                args=training_args,
            )
            
            loss, outputs = trainer.compute_loss(model, mock_batch, return_outputs=True)
            
            assert isinstance(loss, torch.Tensor)
            assert isinstance(outputs, dict)
            assert 'loss' in outputs
            assert 'element_loss' in outputs
            assert 'dist_loss' in outputs
            assert 'noise_loss' in outputs
    
    def test_backward_pass(self, model, mock_batch):
        """Test backward pass works."""
        from transformers import TrainingArguments
        
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=2,
            )
            
            trainer = ReconstructionTrainerWrapper(
                model=model,
                args=training_args,
            )
            
            loss = trainer.compute_loss(model, mock_batch, return_outputs=False)
            loss.backward()
            
            # Check gradients exist
            has_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    has_grad = True
                    break
            assert has_grad


class TestTrainingIntegration:
    """Integration tests for full training setup."""
    
    def test_model_creation(self):
        """Test model creation from arguments."""
        model_args = ModelArguments(
            hidden_dim=128,
            num_layers=3,
        )
        
        encoder = AAEncoder(
            hidden_dim=model_args.hidden_dim,
            num_layers=model_args.num_layers,
            dropout=model_args.dropout,
        )
        
        model = ReconstructionTrainer(
            encoder=encoder,
            num_elements=model_args.num_elements,
            num_dist_bins=model_args.num_dist_bins,
        )
        
        assert model.encoder.hidden_dim == 128
        assert model.num_elements == 119
        assert model.num_dist_bins == 64
    
    def test_collator_creation(self):
        """Test collator creation from arguments."""
        from src.data_loader.aa_dataset import GraphBatchCollator
        
        data_args = DataArguments(
            train_data_path="/dummy/path",
            node_mask_prob=0.2,
            noise_std=0.05,
        )
        
        model_args = ModelArguments()
        
        collator = GraphBatchCollator(
            node_mask_prob=data_args.node_mask_prob,
            noise_std=data_args.noise_std,
            num_dist_bins=model_args.num_dist_bins,
        )
        
        assert collator.node_mask_prob == 0.2
        assert collator.noise_std == 0.05

