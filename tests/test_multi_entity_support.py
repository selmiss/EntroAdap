#!/usr/bin/env python3
# coding=utf-8
"""
Test multi-entity support in data collator and model.

This tests the new functionality where multiple entities can be provided
per sample, with multiple structure tokens in the text.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.models.octopus import Octopus
from src.models.octopus_config import SmallConfig
from src.data_loader.octopus_collator import MultiModalDataCollator, preprocess_multimodal_dataset
from datasets import Dataset, DatasetDict


def create_protein_features(N):
    """Create valid protein node features [N, 7]."""
    return torch.cat([
        torch.randint(0, 119, (N, 1)).float(),
        torch.randint(0, 46, (N, 1)).float(),
        torch.randint(0, 24, (N, 1)).float(),
        torch.randint(0, 27, (N, 1)).float(),
        torch.randint(0, 1000, (N, 1)).float(),
        torch.randint(0, 2, (N, 1)).float(),
        torch.randint(0, 2, (N, 1)).float(),
    ], dim=-1)


class TestMultiEntitySupport:
    """Tests for multi-entity data format."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer with structure tokens."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens": ["<STRUCTURE>"]})
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n'}}{% endfor %}"
        return tokenizer
    
    @pytest.fixture
    def collator(self, tokenizer):
        """Create data collator."""
        return MultiModalDataCollator(tokenizer=tokenizer)
    
    @pytest.fixture
    def small_llm(self):
        """Create a small LLM for testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 4
        config.n_embd = 128
        config.vocab_size = 1000
        llm = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        return llm
    
    @pytest.fixture
    def multimodal_model(self, small_llm):
        """Create integrated multimodal model."""
        model = Octopus(llm_model=small_llm, config=SmallConfig())
        return model
    
    def test_collator_single_entity(self, collator):
        """Test collator with traditional single entity format."""
        N1, E1 = 10, 20
        
        features = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                'attention_mask': [1, 1, 1, 1, 1],
                'labels': [1, 2, 3, 4, 5],
                'modality': 'protein',
                'node_feat': create_protein_features(N1).numpy(),
                'pos': torch.randn(N1, 3).numpy(),
                'edge_index': torch.randint(0, N1, (2, E1)).numpy(),
                'edge_feat_dist': torch.rand(E1, 1).numpy(),
                'patch_position': [2],
            }
        ]
        
        batch = collator(features)
        
        assert 'input_ids' in batch
        assert 'graph_data' in batch
        assert 'patch_positions' in batch
        assert batch['patch_positions'].shape == (1, 1)
        assert batch['patch_positions'][0, 0].item() == 2
    
    def test_collator_multi_entity(self, collator):
        """Test collator with multiple entities (list-wrapped format)."""
        N1, E1 = 10, 20
        N2, E2 = 8, 16
        
        # Multi-entity format: values wrapped in outer list
        features = [
            {
                'input_ids': [1, 2, 3, 4, 5, 6, 7],
                'attention_mask': [1, 1, 1, 1, 1, 1, 1],
                'labels': [1, 2, 3, 4, 5, 6, 7],
                'modality': 'protein',
                # Each field is now a list of entities
                'node_feat': [
                    create_protein_features(N1).numpy(),
                    create_protein_features(N2).numpy()
                ],
                'pos': [
                    torch.randn(N1, 3).numpy(),
                    torch.randn(N2, 3).numpy()
                ],
                'edge_index': [
                    torch.randint(0, N1, (2, E1)).numpy(),
                    torch.randint(0, N2, (2, E2)).numpy()
                ],
                'edge_feat_dist': [
                    torch.rand(E1, 1).numpy(),
                    torch.rand(E2, 1).numpy()
                ],
                'patch_position': [2, 5],  # Two positions for two entities
            }
        ]
        
        batch = collator(features)
        
        assert 'input_ids' in batch
        assert 'graph_data' in batch
        assert 'patch_positions' in batch
        assert batch['patch_positions'].shape == (1, 2)
        assert batch['patch_positions'][0, 0].item() == 2
        assert batch['patch_positions'][0, 1].item() == 5
        
        # Check that entities were merged within the sample
        assert 'node_feat' in batch['graph_data']['value']
        total_nodes = N1 + N2
        assert batch['graph_data']['value']['node_feat'].shape[0] == total_nodes
    
    def test_preprocessing_multi_structure_tokens(self, tokenizer):
        """Test preprocessing finds multiple structure token positions."""
        # Create dataset with multiple structure tokens
        data = {
            "messages": [
                [
                    {"role": "user", "content": "Compare <STRUCTURE> and <STRUCTURE>"},
                    {"role": "assistant", "content": "Done"}
                ]
            ]
        }
        
        train_dataset = Dataset.from_dict(data)
        dataset_dict = DatasetDict({"train": train_dataset})
        
        processed = preprocess_multimodal_dataset(
            dataset_dict,
            tokenizer=tokenizer,
            split='train',
            max_seq_length=128,
        )
        
        example = processed['train'][0]
        assert 'patch_position' in example
        
        # Should return a list with positions or [-1] if not found
        patch_positions = example['patch_position']
        assert isinstance(patch_positions, list)
        print(f"Found patch positions: {patch_positions}")
    
    def test_inject_patches_multiple_positions(self, multimodal_model):
        """Test patch injection at multiple positions via forward pass."""
        B, seq_len = 1, 20
        N1, N2 = 10, 8
        E1, E2 = 20, 16
        
        # Create merged graph (simulating collator output)
        total_nodes = N1 + N2
        node_feat = torch.cat([create_protein_features(N1), create_protein_features(N2)], dim=0)
        pos = torch.randn(total_nodes, 3)
        
        # Edge indices with offsets
        edge_index_1 = torch.randint(0, N1, (2, E1))
        edge_index_2 = torch.randint(0, N2, (2, E2)) + N1
        edge_index = torch.cat([edge_index_1, edge_index_2], dim=1)
        edge_feat_dist = torch.rand(E1 + E2, 1)
        
        input_ids = torch.randint(0, 1000, (B, seq_len))
        
        graph_data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_feat_dist': edge_feat_dist,
                'edge_index': edge_index,
                'pos': pos,
            }
        }
        
        batch_assignment = torch.zeros(total_nodes, dtype=torch.long)
        instr_positions = torch.tensor([[0, 1, -1]], dtype=torch.long)
        # Multiple patch positions [B, 2]
        patch_positions = torch.tensor([[3, 10]], dtype=torch.long)
        
        outputs = multimodal_model(
            input_ids=input_ids,
            graph_data=graph_data,
            batch=batch_assignment,
            instr_positions=instr_positions,
            patch_positions=patch_positions,
        )
        
        # Output sequence should be seq_len + k_max * num_positions
        k_max = multimodal_model.config_octopus.patching.k_max
        num_positions = patch_positions.shape[1]
        expected_len = seq_len + k_max * num_positions
        assert outputs.logits.shape == (B, expected_len, 1000)
    
    def test_batch_with_mixed_entity_counts(self, collator):
        """Test batch where some samples have multiple entities and some have single."""
        N1, E1 = 10, 20
        N2, E2 = 8, 16
        N3, E3 = 12, 24
        
        features = [
            # Sample 1: Two entities
            {
                'input_ids': [1, 2, 3, 4, 5, 6],
                'attention_mask': [1, 1, 1, 1, 1, 1],
                'labels': [1, 2, 3, 4, 5, 6],
                'modality': 'protein',
                'node_feat': [
                    create_protein_features(N1).numpy(),
                    create_protein_features(N2).numpy()
                ],
                'pos': [
                    torch.randn(N1, 3).numpy(),
                    torch.randn(N2, 3).numpy()
                ],
                'edge_index': [
                    torch.randint(0, N1, (2, E1)).numpy(),
                    torch.randint(0, N2, (2, E2)).numpy()
                ],
                'edge_feat_dist': [
                    torch.rand(E1, 1).numpy(),
                    torch.rand(E2, 1).numpy()
                ],
                'patch_position': [2, 4],
            },
            # Sample 2: Single entity
            {
                'input_ids': [1, 2, 3, 4, 5],
                'attention_mask': [1, 1, 1, 1, 1],
                'labels': [1, 2, 3, 4, 5],
                'modality': 'protein',
                'node_feat': create_protein_features(N3).numpy(),
                'pos': torch.randn(N3, 3).numpy(),
                'edge_index': torch.randint(0, N3, (2, E3)).numpy(),
                'edge_feat_dist': torch.rand(E3, 1).numpy(),
                'patch_position': [1],
            }
        ]
        
        batch = collator(features)
        
        assert 'patch_positions' in batch
        # Should be padded to same length (2 positions)
        assert batch['patch_positions'].shape == (2, 2)
        # First sample has two positions
        assert batch['patch_positions'][0, 0].item() >= 0
        assert batch['patch_positions'][0, 1].item() >= 0
        # Second sample has one position, second is padded with -1
        assert batch['patch_positions'][1, 0].item() >= 0
        assert batch['patch_positions'][1, 1].item() == -1
    
    def test_forward_multi_entity(self, multimodal_model):
        """Test full forward pass with multi-entity input."""
        B, seq_len = 1, 15
        N1, N2 = 10, 8
        E1, E2 = 20, 16
        
        # Create merged graph (simulating collator output)
        total_nodes = N1 + N2
        node_feat = torch.cat([create_protein_features(N1), create_protein_features(N2)], dim=0)
        pos = torch.randn(total_nodes, 3)
        
        # Edge indices with offsets
        edge_index_1 = torch.randint(0, N1, (2, E1))
        edge_index_2 = torch.randint(0, N2, (2, E2)) + N1  # Offset for second entity
        edge_index = torch.cat([edge_index_1, edge_index_2], dim=1)
        edge_feat_dist = torch.rand(E1 + E2, 1)
        
        input_ids = torch.randint(0, 1000, (B, seq_len))
        
        graph_data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_feat_dist': edge_feat_dist,
                'edge_index': edge_index,
                'pos': pos,
            }
        }
        
        # Batch assignment: first N1 nodes belong to entity 0, next N2 to entity 0 (same sample)
        batch_assignment = torch.zeros(total_nodes, dtype=torch.long)
        
        instr_positions = torch.tensor([[0, 1, -1]], dtype=torch.long)
        # Multiple patch positions
        patch_positions = torch.tensor([[3, 8]], dtype=torch.long)
        
        outputs = multimodal_model(
            input_ids=input_ids,
            graph_data=graph_data,
            batch=batch_assignment,
            instr_positions=instr_positions,
            patch_positions=patch_positions,
        )
        
        # Output sequence should be seq_len + k_max * num_positions
        k_max = multimodal_model.config_octopus.patching.k_max
        num_positions = patch_positions.shape[1]
        expected_len = seq_len + k_max * num_positions
        assert outputs.logits.shape == (B, expected_len, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

