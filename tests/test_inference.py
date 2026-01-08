"""
Tests for inference module.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
import pandas as pd

from src.runner.inference import (
    InferenceArguments,
    prepare_graph_data,
    prepare_messages_for_inference,
    load_input_data,
)


class TestInferenceArguments:
    """Test inference arguments."""
    
    def test_default_arguments(self):
        """Test default argument values."""
        args = InferenceArguments(
            checkpoint_path="./test_checkpoint",
            input_file="test.parquet",
        )
        assert args.max_new_tokens == 256
        assert args.temperature == 0.7
        assert args.do_sample is True
        assert args.batch_size == 1
    
    def test_custom_arguments(self):
        """Test custom argument values."""
        args = InferenceArguments(
            checkpoint_path="./checkpoint",
            input_file="input.jsonl",
            output_file="output.jsonl",
            max_new_tokens=128,
            temperature=0.5,
            top_p=0.95,
            do_sample=False,
        )
        assert args.max_new_tokens == 128
        assert args.temperature == 0.5
        assert args.top_p == 0.95
        assert args.do_sample is False


class TestPrepareGraphData:
    """Test graph data preparation."""
    
    def test_prepare_molecule_graph(self):
        """Test preparing molecule graph data."""
        sample = {
            "modality": "molecule",
            "node_feat": [[1, 0, 3, 5, 2, 0, 1, 0, 0]] * 10,
            "pos": [[0.0, 0.0, 0.0]] * 10,
            "edge_index": [[0, 1, 2], [1, 2, 3]],
        }
        
        graph_data = prepare_graph_data(sample)
        
        assert graph_data is not None
        assert graph_data["modality"] == "molecule"
        assert "node_feat" in graph_data["value"]
        assert "pos" in graph_data["value"]
        assert "edge_index" in graph_data["value"]
        
        # Check tensor types
        assert isinstance(graph_data["value"]["node_feat"], torch.Tensor)
        assert isinstance(graph_data["value"]["pos"], torch.Tensor)
        assert isinstance(graph_data["value"]["edge_index"], torch.Tensor)
    
    def test_prepare_protein_graph(self):
        """Test preparing protein graph data."""
        sample = {
            "modality": "protein",
            "node_feat": [[6, 1, 0, 0, 100, 1, 0]] * 20,
            "pos": [[i, i, i] for i in range(20)],
            "edge_index": [[i, i+1] for i in range(19)],
            "edge_feat_dist": [[1.5]] * 19,
        }
        
        graph_data = prepare_graph_data(sample)
        
        assert graph_data is not None
        assert graph_data["modality"] == "protein"
        assert "edge_feat_dist" in graph_data["value"]
    
    def test_no_graph_data(self):
        """Test sample without graph data."""
        sample = {
            "modality": "text",
            "node_feat": None,
        }
        
        graph_data = prepare_graph_data(sample)
        assert graph_data is None
    
    def test_edge_index_transpose(self):
        """Test edge_index is properly transposed to [2, E]."""
        # Input as [E, 2]
        sample = {
            "modality": "molecule",
            "node_feat": [[1, 0, 3, 5, 2, 0, 1, 0, 0]] * 5,
            "pos": [[0.0, 0.0, 0.0]] * 5,
            "edge_index": [[0, 1], [1, 2], [2, 3]],  # [E, 2] format
        }
        
        graph_data = prepare_graph_data(sample)
        edge_index = graph_data["value"]["edge_index"]
        
        # Should be transposed to [2, E]
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 3


class TestPrepareMessages:
    """Test message preparation for inference."""
    
    def test_user_query(self):
        """Test when last message is from user."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is this?"},
        ]
        
        msgs, label, is_user = prepare_messages_for_inference(messages)
        
        assert is_user is True
        assert label is None
        assert len(msgs) == 2
    
    def test_assistant_response(self):
        """Test when last message is from assistant."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is this?"},
            {"role": "assistant", "content": "This is a protein."},
        ]
        
        msgs, label, is_user = prepare_messages_for_inference(messages)
        
        assert is_user is False
        assert label == "This is a protein."
        assert len(msgs) == 2  # Excludes last assistant message
    
    def test_empty_messages(self):
        """Test with empty messages."""
        messages = []
        
        msgs, label, is_user = prepare_messages_for_inference(messages)
        
        assert is_user is True
        assert label is None
        assert len(msgs) == 0


class TestLoadInputData:
    """Test input data loading."""
    
    def test_load_parquet(self):
        """Test loading parquet file."""
        # Create temporary parquet file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name
        
        try:
            # Create test data
            df = pd.DataFrame({
                "modality": ["molecule", "protein"],
                "messages": [
                    [{"role": "user", "content": "Test 1"}],
                    [{"role": "user", "content": "Test 2"}],
                ],
            })
            df.to_parquet(temp_path)
            
            # Load data
            loaded_df = load_input_data(temp_path)
            
            assert len(loaded_df) == 2
            assert "modality" in loaded_df.columns
            assert "messages" in loaded_df.columns
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_jsonl(self):
        """Test loading jsonl file."""
        # Create temporary jsonl file
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode='w') as f:
            temp_path = f.name
            f.write(json.dumps({"modality": "molecule", "messages": []}) + '\n')
            f.write(json.dumps({"modality": "protein", "messages": []}) + '\n')
        
        try:
            # Load data
            loaded_df = load_input_data(temp_path)
            
            assert len(loaded_df) == 2
            assert "modality" in loaded_df.columns
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_max_samples(self):
        """Test max_samples parameter."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode='w') as f:
            temp_path = f.name
            for i in range(10):
                f.write(json.dumps({"modality": "test", "idx": i}) + '\n')
        
        try:
            # Load with limit
            loaded_df = load_input_data(temp_path, max_samples=5)
            
            assert len(loaded_df) == 5
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_unsupported_format(self):
        """Test error on unsupported file format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_input_data("test.txt")


class TestIntegration:
    """Integration tests for inference workflow."""
    
    def test_full_workflow_structure(self):
        """Test that all components work together structurally."""
        # Create mock sample
        sample = {
            "modality": "molecule",
            "node_feat": [[1, 0, 3, 5, 2, 0, 1, 0, 0]] * 5,
            "pos": [[float(i), float(i), float(i)] for i in range(5)],
            "edge_index": [[0, 1, 2], [1, 2, 3]],
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Describe this molecule: <STRUCTURE>"},
            ],
        }
        
        # Test graph preparation
        graph_data = prepare_graph_data(sample)
        assert graph_data is not None
        assert graph_data["modality"] == "molecule"
        
        # Test message preparation
        messages = sample["messages"]
        msgs, label, is_user = prepare_messages_for_inference(messages)
        assert is_user is True
        assert len(msgs) == 2
        
        # Test that tensors are proper shape
        node_feat = graph_data["value"]["node_feat"]
        pos = graph_data["value"]["pos"]
        edge_index = graph_data["value"]["edge_index"]
        
        assert node_feat.shape == (5, 9)
        assert pos.shape == (5, 3)
        assert edge_index.shape[0] == 2

