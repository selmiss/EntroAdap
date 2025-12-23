#!/usr/bin/env python3
"""
Test script for end-to-end multimodal SFT data loading pipeline.

Tests:
1. MultiModalSFTDataset with graph loading
2. preprocess_multimodal_sft_dataset
3. MultiModalDataCollator batching
4. Output compatibility with Octopus
"""

import torch
import tempfile
import json
from pathlib import Path

from transformers import AutoTokenizer
from src.data_loader import MultiModalSFTDataset, preprocess_multimodal_sft_dataset, MultiModalDataCollator
from src.models import Octopus
from src.models.octopus_config import SmallConfig


def create_sample_jsonl(path: Path, num_samples: int = 5):
    """Create a sample JSONL dataset with text and structure info."""
    # For testing, we'll use mock inline graph data
    samples = []
    for i in range(num_samples):
        # Protein node features (7):
        # [atomic_number(0-118), atom_name(0-45), residue(0-23), chain(0-26),
        #  residue_id(continuous), is_backbone(0/1), is_ca(0/1)]
        n = 10 + i
        atomic_number = torch.randint(0, 119, (n, 1))
        atom_name = torch.randint(0, 46, (n, 1))
        residue = torch.randint(0, 24, (n, 1))
        chain = torch.randint(0, 27, (n, 1))
        residue_id = torch.arange(n, dtype=torch.float32).unsqueeze(1)  # continuous
        is_backbone = torch.randint(0, 2, (n, 1))
        is_ca = torch.randint(0, 2, (n, 1))
        node_feat = torch.cat(
            [atomic_number, atom_name, residue, chain, residue_id, is_backbone, is_ca],
            dim=1,
        )

        sample = {
            "messages": [
                {"role": "system", "content": "You are a protein structure analyzer."},
                {"role": "user", "content": f"Analyze structure #{i}: <STRUCTURE>"},
                {"role": "assistant", "content": f"This is a sample protein with {10+i} residues."}
            ],
            "structure": {
                "type": "inline",
                "modality": "protein",
                "value": {
                    "modality": "protein",
                    "value": {
                        "node_feat": node_feat.tolist(),
                        "pos": torch.randn(n, 3).tolist(),
                        "edge_index": torch.randint(0, n, (2, n * 3)).tolist(),
                        "edge_attr": torch.rand(n * 3, 1).tolist(),
                    }
                }
            }
        }
        samples.append(sample)
    
    with open(path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


def test_dataset_loading():
    """Test 1: Dataset loading and graph extraction."""
    print("\n=== Test 1: Dataset Loading ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_data.jsonl"
        create_sample_jsonl(dataset_path, num_samples=3)
        
        # Load dataset
        dataset = MultiModalSFTDataset(str(dataset_path))
        
        print(f"✓ Dataset loaded: {len(dataset)} examples")
        
        # Get first example
        example = dataset[0]
        assert 'messages' in example
        assert 'graph_data' in example
        assert example['graph_data']['modality'] == 'protein'
        
        print(f"✓ Example structure: messages={len(example['messages'])}, graph modality={example['graph_data']['modality']}")
        print(f"  Graph nodes: {example['graph_data']['value']['node_feat'].shape}")


def test_preprocessing():
    """Test 2: Tokenization and preprocessing."""
    print("\n=== Test 2: Preprocessing ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_data.jsonl"
        create_sample_jsonl(dataset_path, num_samples=3)
        
        # Load dataset
        dataset = MultiModalSFTDataset(str(dataset_path))
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # Add <STRUCTURE> token for patch injection
        tokenizer.add_special_tokens({"additional_special_tokens": ["<STRUCTURE>"]})
        # Set chat template
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n'}}{% endfor %}"
        
        # Preprocess
        processed = preprocess_multimodal_sft_dataset(
            dataset,
            tokenizer=tokenizer,
            max_seq_length=512,
        )
        
        print(f"✓ Dataset preprocessed")
        
        # Check first example
        example = processed[0]
        assert 'input_ids' in example
        assert 'labels' in example
        assert 'instr_len' in example
        assert 'graph_data' in example
        assert 'patch_position' in example
        
        print(f"✓ Preprocessed example: input_ids len={len(example['input_ids'])}, instr_len={example['instr_len']}")
        print(f"  patch_position: {example['patch_position']}")


def test_collator():
    """Test 3: Batch collation."""
    print("\n=== Test 3: Batch Collation ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_data.jsonl"
        create_sample_jsonl(dataset_path, num_samples=5)
        
        dataset = MultiModalSFTDataset(str(dataset_path))
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # Add <STRUCTURE> token
        tokenizer.add_special_tokens({"additional_special_tokens": ["<STRUCTURE>"]})
        # Set chat template
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n'}}{% endfor %}"
        
        processed = preprocess_multimodal_sft_dataset(
            dataset,
            tokenizer=tokenizer,
            max_seq_length=512,
        )
        
        # Create collator
        collator = MultiModalDataCollator(tokenizer=tokenizer)
        
        # Collate batch
        batch_size = 3
        features = [processed[i] for i in range(batch_size)]
        batch = collator(features)
        
        print(f"✓ Batch created: {batch_size} examples")
        print(f"  Keys: {list(batch.keys())}")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        
        # Check graph data
        assert 'graph_data' in batch
        assert 'batch' in batch
        assert 'instr_positions' in batch
        
        print(f"  graph_data modality: {batch['graph_data']['modality']}")
        print(f"  batch tensor (node assignment): {batch['batch'].shape}")
        print(f"  instr_positions shape: {batch['instr_positions'].shape}")
        
        # Check patch_positions if <STRUCTURE> token was used
        if 'patch_positions' in batch:
            print(f"  patch_positions shape: {batch['patch_positions'].shape}")
            assert batch['patch_positions'].shape[0] == batch_size
        
        # Verify node count matches batch tensor
        num_nodes = batch['graph_data']['value']['node_feat'].shape[0]
        num_graphs = batch['batch'].max().item() + 1
        print(f"  Total nodes: {num_nodes}, Num graphs in batch: {num_graphs}")
        
        assert num_graphs == batch_size or num_graphs <= batch_size


def test_model_forward():
    """Test 4: Forward pass through Octopus."""
    print("\n=== Test 4: Model Forward Pass ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_data.jsonl"
        create_sample_jsonl(dataset_path, num_samples=2)
        
        dataset = MultiModalSFTDataset(str(dataset_path))
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # Add <STRUCTURE> token
        tokenizer.add_special_tokens({"additional_special_tokens": ["<STRUCTURE>"]})
        # Set chat template
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n'}}{% endfor %}"
        
        processed = preprocess_multimodal_sft_dataset(
            dataset,
            tokenizer=tokenizer,
            max_seq_length=128,
        )
        
        collator = MultiModalDataCollator(tokenizer=tokenizer)
        batch = collator([processed[i] for i in range(2)])
        
        # Create small model
        from transformers import AutoConfig, AutoModelForCausalLM
        llm_config = AutoConfig.from_pretrained("gpt2")
        llm_config.n_layer = 2
        llm_config.n_head = 2
        llm_config.n_embd = 128
        llm_config.vocab_size = len(tokenizer)  # Use actual vocab size after adding special tokens
        llm = AutoModelForCausalLM.from_config(llm_config, attn_implementation="eager")
        
        model = Octopus(llm_model=llm, config=SmallConfig())
        model.eval()
        
        print(f"✓ Model created")
        
        # Forward pass with patch_positions if available
        forward_kwargs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels'],
            'graph_data': batch['graph_data'],
            'batch': batch['batch'],
            'instr_positions': batch['instr_positions'],
        }
        if 'patch_positions' in batch:
            forward_kwargs['patch_positions'] = batch['patch_positions']
            print(f"  Using patch_positions: {batch['patch_positions'].tolist()}")
        
        with torch.no_grad():
            outputs = model(**forward_kwargs)
        
        print(f"✓ Forward pass successful")
        print(f"  Loss: {outputs.loss.item():.4f}")
        print(f"  Logits shape: {outputs.logits.shape}")
        
        # Verify output shape
        # Note: Sequence length increases by k_max patches
        k_max = model.config_octopus.patching.k_max
        expected_seq_len = batch['input_ids'].shape[1] + k_max
        assert outputs.logits.shape == (2, expected_seq_len, len(tokenizer))
        print(f"  Expected seq len (with patches): {expected_seq_len}")
        print(f"  Actual seq len: {outputs.logits.shape[1]}")


def main():
    """Run all tests."""
    print("="*60)
    print("Multi-modal SFT Data Loading Pipeline Tests")
    print("="*60)
    
    try:
        test_dataset_loading()
        test_preprocessing()
        test_collator()
        test_model_forward()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

