#!/usr/bin/env python3
# coding=utf-8
"""
Tests for multi-modal data collator.
"""

import pytest
import torch
from transformers import AutoTokenizer

from src.data_loader.octopus_collator import MultiModalDataCollator


class TestMultiModalDataCollator:
    """Tests for MultiModalDataCollator."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @pytest.fixture
    def collator(self, tokenizer):
        """Create a collator for testing."""
        return MultiModalDataCollator(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )
    
    def test_collate_text_only(self, collator):
        """Test collating text-only examples."""
        features = [
            {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]},
            {"input_ids": [6, 7, 8], "attention_mask": [1, 1, 1]},
        ]
        
        batch = collator(features)
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[1] == 5  # Padded to longest
    
    def test_collate_with_labels(self, collator):
        """Test collating examples with labels."""
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5],
            },
            {
                "input_ids": [6, 7, 8],
                "attention_mask": [1, 1, 1],
                "labels": [6, 7, 8],
            },
        ]
        
        batch = collator(features)
        
        assert "labels" in batch
        assert batch["labels"].shape == batch["input_ids"].shape
    
    def test_collate_with_graph_data(self, collator):
        """Test collating examples with graph data."""
        # Create minimal graph data for protein modality
        graph_1 = {
            'modality': 'protein',
            'value': {
                'node_feat': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],  # 3 nodes
                'pos': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                'edge_index': [[0, 1], [1, 2]],  # 2 edges
            }
        }
        graph_2 = {
            'modality': 'protein',
            'value': {
                'node_feat': [[0.7, 0.8], [0.9, 1.0]],  # 2 nodes
                'pos': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                'edge_index': [[0, 1]],  # 1 edge
            }
        }
        
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "graph_data": graph_1,
            },
            {
                "input_ids": [6, 7, 8],
                "attention_mask": [1, 1, 1],
                "graph_data": graph_2,
            },
        ]
        
        batch = collator(features)
        
        assert "graph_data" in batch
        assert "batch" in batch
        assert batch["graph_data"]["modality"] == "protein"
        # Check that graph data has expected keys
        assert "node_feat" in batch["graph_data"]["value"]
        assert "pos" in batch["graph_data"]["value"]
        assert "edge_index" in batch["graph_data"]["value"]
    
    def test_collate_with_instr_positions(self, collator):
        """Test collating examples with instruction positions."""
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "instr_positions": [0, 1, 2],
            },
            {
                "input_ids": [6, 7, 8],
                "attention_mask": [1, 1, 1],
                "instr_positions": [0, 1, 2, 3, 4],
            },
        ]
        
        batch = collator(features)
        
        assert "instr_positions" in batch
        assert batch["instr_positions"].shape[0] == 2
        assert batch["instr_positions"].shape[1] == 5  # Padded to longest
    
    def test_collate_with_patch_positions(self, collator):
        """Test collating examples with patch positions."""
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "patch_position": 2,
            },
            {
                "input_ids": [6, 7, 8],
                "attention_mask": [1, 1, 1],
                "patch_position": 1,
            },
        ]
        
        batch = collator(features)
        
        assert "patch_positions" in batch
        assert batch["patch_positions"].shape[0] == 2
        assert batch["patch_positions"].shape[1] == 1  # Single position per sample
        assert batch["patch_positions"][0, 0] == 2
        assert batch["patch_positions"][1, 0] == 1
    
    def test_collate_mixed_examples(self, collator):
        """Test collating examples where some have graph data and some don't."""
        graph_data = {
            'modality': 'protein',
            'value': {
                'node_feat': [[0.1, 0.2], [0.3, 0.4]],
                'pos': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                'edge_index': [[0, 1]],
            }
        }
        
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "graph_data": graph_data,
                "patch_position": 2,
            },
            {
                "input_ids": [6, 7, 8],
                "attention_mask": [1, 1, 1],
                # No graph data
            },
        ]
        
        batch = collator(features)
        
        assert "input_ids" in batch
        assert "graph_data" in batch
        assert "_graph_indices" in batch
        # Only first example has graph data
        assert batch["_graph_indices"] == [0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
