#!/usr/bin/env python3
# coding=utf-8
"""
Tests for multimodal data loading and processing.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from transformers import AutoTokenizer

from src.data_loader import MultiModalDataCollator
from utils.data import get_dataset
from src.configs import ScriptArguments


class TestMultiModalDataLoading:
    """Tests for loading multimodal data from JSONL files."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @pytest.fixture
    def mock_multimodal_jsonl(self):
        """Create a temporary JSONL file with multimodal data."""
        import random
        
        # Generate random embeddings
        modality_dim = 256
        text_dim = 768
        
        data = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "The answer is 4."}
                ],
                "modality_embeddings": [[random.gauss(0, 0.1) for _ in range(modality_dim)] for _ in range(4)],
                "kv_embeddings": [[random.gauss(0, 0.1) for _ in range(text_dim)] for _ in range(5)]
            },
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ],
                "modality_embeddings": [[random.gauss(0, 0.1) for _ in range(modality_dim)] for _ in range(5)],
                "kv_embeddings": [[random.gauss(0, 0.1) for _ in range(text_dim)] for _ in range(3)]
            },
        ]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_load_multimodal_jsonl(self, mock_multimodal_jsonl):
        """Test loading multimodal data from JSONL file."""
        script_args = ScriptArguments(dataset_name=mock_multimodal_jsonl)
        
        dataset = get_dataset(script_args)
        
        assert 'train' in dataset
        assert len(dataset['train']) == 2
        
        # Check first example
        example = dataset['train'][0]
        assert 'messages' in example
        assert 'modality_embeddings' in example
        assert 'kv_embeddings' in example
        
        # Check data types
        assert isinstance(example['messages'], list)
        assert isinstance(example['modality_embeddings'], list)
        assert isinstance(example['kv_embeddings'], list)
    
    def test_multimodal_data_has_required_fields(self, mock_multimodal_jsonl):
        """Test that loaded multimodal data has all required fields."""
        script_args = ScriptArguments(dataset_name=mock_multimodal_jsonl)
        dataset = get_dataset(script_args)
        
        required_fields = ['messages', 'modality_embeddings', 'kv_embeddings']
        
        for example in dataset['train']:
            for field in required_fields:
                assert field in example, f"Missing required field: {field}"
    
    def test_multimodal_collator_with_data(self, tokenizer, mock_multimodal_jsonl):
        """Test MultiModalDataCollator with real multimodal data."""
        # Load data
        script_args = ScriptArguments(dataset_name=mock_multimodal_jsonl)
        dataset = get_dataset(script_args)
        
        # Prepare features (simulate preprocessing)
        features = []
        for example in dataset['train']:
            # Tokenize messages (simplified)
            text = " ".join([msg['content'] for msg in example['messages']])
            tokenized = tokenizer(text, truncation=True, max_length=50)
            
            features.append({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': tokenized['input_ids'].copy(),
                'modality_embeddings': example['modality_embeddings'],
                'kv_embeddings': example['kv_embeddings'],
            })
        
        # Collate
        collator = MultiModalDataCollator(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )
        batch = collator(features)
        
        # Check batch structure
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        assert 'modality_embeddings' in batch
        assert 'modality_attention_mask' in batch
        assert 'kv_embeddings' in batch
        assert 'text_attention_mask' in batch
        
        # Check shapes
        batch_size = len(features)
        assert batch['input_ids'].shape[0] == batch_size
        assert batch['modality_embeddings'].shape[0] == batch_size
        assert batch['kv_embeddings'].shape[0] == batch_size


class TestMultiModalCollator:
    """Tests for MultiModalDataCollator."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @pytest.fixture
    def collator(self, tokenizer):
        """Create a MultiModalDataCollator."""
        return MultiModalDataCollator(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
            include_modality_tokens=True,
        )
    
    def test_collate_with_modality_embeddings(self, collator):
        """Test collating examples with modality embeddings."""
        embed_dim = 256
        text_dim = 768
        features = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                'attention_mask': [1, 1, 1, 1, 1],
                'labels': [1, 2, 3, 4, 5],
                'modality_embeddings': [[0.1] * embed_dim for _ in range(3)],
                'kv_embeddings': [[0.2] * text_dim for _ in range(4)],
            },
            {
                'input_ids': [6, 7, 8],
                'attention_mask': [1, 1, 1],
                'labels': [6, 7, 8],
                'modality_embeddings': [[0.3] * embed_dim for _ in range(2)],
                'kv_embeddings': [[0.4] * text_dim for _ in range(2)],
            },
        ]
        
        batch = collator(features)
        
        # Check that modality embeddings are padded
        assert batch['modality_embeddings'].shape == (2, 3, embed_dim)  # Max length is 3
        assert batch['modality_attention_mask'].shape == (2, 3)
        
        # Check padding
        assert batch['modality_attention_mask'][1, 2] == 0  # Masked
    
    def test_collate_without_modality_embeddings(self, collator):
        """Test collating examples without modality embeddings."""
        features = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                'attention_mask': [1, 1, 1, 1, 1],
                'labels': [1, 2, 3, 4, 5],
            },
            {
                'input_ids': [6, 7, 8],
                'attention_mask': [1, 1, 1],
                'labels': [6, 7, 8],
            },
        ]
        
        batch = collator(features)
        
        # Should not have modality fields
        assert 'modality_embeddings' not in batch or batch['modality_embeddings'].numel() == 0
    
    def test_collate_varying_lengths(self, collator):
        """Test collating examples with varying sequence lengths."""
        embed_dim = 256
        text_dim = 768
        features = [
            {
                'input_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'labels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'modality_embeddings': [[0.1] * embed_dim for _ in range(5)],
                'kv_embeddings': [[0.2] * text_dim for _ in range(3)],
            },
            {
                'input_ids': [11, 12],
                'labels': [11, 12],
                'modality_embeddings': [[0.3] * embed_dim for _ in range(1)],
                'kv_embeddings': [[0.4] * text_dim for _ in range(4)],
            },
        ]
        
        batch = collator(features)
        
        # All sequences should be padded to max length
        assert batch['input_ids'].shape[1] == 10  # Max text length
        assert batch['modality_embeddings'].shape[1] == 5  # Max modality length
        assert batch['kv_embeddings'].shape[1] == 4  # Max cross-attn length
        
        # Check padding values
        assert batch['labels'][1, 2] == -100  # Padded label


class TestDataFormatValidation:
    """Tests to validate data format for training."""
    
    def test_mock_train_jsonl_format(self):
        """Test that data/mock_train.jsonl has the correct format."""
        data_path = Path(__file__).parent.parent / "data" / "mock_train.jsonl"
        
        if not data_path.exists():
            pytest.skip("mock_train.jsonl not found")
        
        with open(data_path) as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {line_num} is not valid JSON: {e}")
                
                # Check required fields
                assert 'messages' in example, f"Line {line_num} missing 'messages'"
                assert isinstance(example['messages'], list), f"Line {line_num} 'messages' not a list"
                
                # Check message format
                for msg in example['messages']:
                    assert 'role' in msg, f"Line {line_num} message missing 'role'"
                    assert 'content' in msg, f"Line {line_num} message missing 'content'"
                
                # Check multimodal fields (optional but recommended)
                if 'modality_embeddings' in example:
                    assert isinstance(example['modality_embeddings'], list), \
                        f"Line {line_num} 'modality_embeddings' not a list"
                    assert all(isinstance(e, list) for e in example['modality_embeddings']), \
                        f"Line {line_num} 'modality_embeddings' should contain lists of floats"
                
                if 'kv_embeddings' in example:
                    assert isinstance(example['kv_embeddings'], list), \
                        f"Line {line_num} 'kv_embeddings' not a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
