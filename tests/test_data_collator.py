#!/usr/bin/env python3
# coding=utf-8
"""
Tests for multi-modal data collator.
"""

import pytest
import torch
from transformers import AutoTokenizer

from src.data_loader.multimodal_collator import MultiModalDataCollator


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
            include_modality_tokens=True,
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
    
    def test_collate_with_modality_embeddings(self, collator):
        """Test collating examples with modality embeddings."""
        embed_dim = 256
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "modality_embeddings": [[0.1] * embed_dim for _ in range(4)],
            },
            {
                "input_ids": [6, 7, 8],
                "attention_mask": [1, 1, 1],
                "modality_embeddings": [[0.2] * embed_dim for _ in range(2)],
            },
        ]
        
        batch = collator(features)
        
        assert "modality_embeddings" in batch
        assert "modality_attention_mask" in batch
        assert batch["modality_embeddings"].shape[0] == 2
        assert batch["modality_embeddings"].shape[1] == 4  # Padded to longest
        assert batch["modality_embeddings"].shape[2] == embed_dim
        
        # Check attention mask
        assert batch["modality_attention_mask"][0].sum() == 4  # All valid
        assert batch["modality_attention_mask"][1].sum() == 2  # 2 valid, 2 padding
    
    def test_collate_with_kv_embeddings(self, collator):
        """Test collating examples with cross-attention text embeddings."""
        embed_dim = 768
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "kv_embeddings": [[0.3] * embed_dim for _ in range(3)],
            },
            {
                "input_ids": [6, 7, 8],
                "attention_mask": [1, 1, 1],
                "kv_embeddings": [[0.4] * embed_dim for _ in range(5)],
            },
        ]
        
        batch = collator(features)
        
        assert "kv_embeddings" in batch
        assert "text_attention_mask" in batch
        assert batch["kv_embeddings"].shape[0] == 2
        assert batch["kv_embeddings"].shape[1] == 5  # Padded to longest
        assert batch["kv_embeddings"].shape[2] == embed_dim
    
    def test_collate_with_modality_positions(self, collator):
        """Test collating examples with modality positions."""
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "modality_positions": [1, 2, 3],
            },
            {
                "input_ids": [6, 7, 8],
                "attention_mask": [1, 1, 1],
                "modality_positions": [0, 1],
            },
        ]
        
        batch = collator(features)
        
        assert "modality_positions" in batch
        assert batch["modality_positions"].shape[0] == 2
        assert batch["modality_positions"].shape[1] == 3  # Padded to longest
        
        # Check padding value is -1
        assert batch["modality_positions"][1, 2] == -1
    
    def test_collate_mixed_examples(self, collator):
        """Test collating examples where some have modality and some don't."""
        embed_dim = 256
        text_embed_dim = 768
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "modality_embeddings": [[0.1] * embed_dim for _ in range(3)],
                "kv_embeddings": [[0.2] * text_embed_dim for _ in range(2)],
            },
            {
                "input_ids": [6, 7, 8],
                "attention_mask": [1, 1, 1],
                # No modality embeddings
            },
        ]
        
        batch = collator(features)
        
        assert "input_ids" in batch
        assert "modality_embeddings" in batch
        assert "modality_attention_mask" in batch
        
        # Second example should have all padding for modality
        assert batch["modality_attention_mask"][1].sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
