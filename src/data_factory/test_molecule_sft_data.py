"""
Test script to verify the generated SFT data works with the training pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_loader.octopus_sft_dataset import MultiModalSFTDataset
import torch

def test_dataset_loading():
    """Test basic dataset loading."""
    print("=== Test 1: Dataset Loading ===")
    dataset = MultiModalSFTDataset(
        dataset_path='data/molecule_sft_train.parquet',
        use_combined_parquet=True
    )
    
    assert len(dataset) == 49, f"Expected 49 examples, got {len(dataset)}"
    print(f"✓ Loaded {len(dataset)} examples")
    
    return dataset


def test_data_format(dataset):
    """Test data format compatibility."""
    print("\n=== Test 2: Data Format ===")
    
    example = dataset[0]
    
    # Check required keys
    required_keys = ['messages', 'graph_data', 'structure_token', 'smiles']
    for key in required_keys:
        assert key in example, f"Missing key: {key}"
    print(f"✓ All required keys present: {required_keys}")
    
    # Check messages format
    messages = example['messages']
    assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    assert messages[2]['role'] == 'assistant'
    assert '<STRUCTURE>' in messages[1]['content']
    print(f"✓ Messages format correct (3 messages with correct roles)")
    
    # Check graph data format
    graph_data = example['graph_data']
    assert graph_data['modality'] == 'molecule'
    assert 'value' in graph_data
    
    value = graph_data['value']
    required_graph_keys = ['node_feat', 'pos', 'edge_index']
    for key in required_graph_keys:
        assert key in value, f"Missing graph key: {key}"
        assert isinstance(value[key], torch.Tensor), f"{key} should be a tensor"
    print(f"✓ Graph data format correct")
    
    # Check tensor shapes
    node_feat = value['node_feat']
    pos = value['pos']
    edge_index = value['edge_index']
    
    num_nodes = node_feat.shape[0]
    assert pos.shape[0] == num_nodes, "Position count mismatch"
    assert node_feat.shape[1] == 9, f"Expected 9 node features, got {node_feat.shape[1]}"
    assert pos.shape[1] == 3, f"Expected 3D positions, got {pos.shape[1]}"
    assert edge_index.shape[0] == 2, f"Expected 2-row edge_index, got {edge_index.shape[0]}"
    
    print(f"✓ Tensor shapes: nodes={num_nodes}, node_feat={node_feat.shape}, pos={pos.shape}, edges={edge_index.shape[1]}")
    
    return example


def test_multiple_examples(dataset):
    """Test loading multiple examples."""
    print("\n=== Test 3: Multiple Examples ===")
    
    # Test first 5 examples
    for i in range(min(5, len(dataset))):
        example = dataset[i]
        assert 'messages' in example
        assert 'graph_data' in example
        assert '<STRUCTURE>' in example['messages'][1]['content']
    
    print(f"✓ Successfully loaded first 5 examples")
    
    # Check diversity in task types
    user_messages = [dataset[i]['messages'][1]['content'][:100] for i in range(min(8, len(dataset)))]
    unique_starts = len(set(tuple(msg.split()[0:5]) for msg in user_messages if msg))
    print(f"✓ Found {unique_starts} unique instruction patterns (diverse tasks)")


def test_smiles_correspondence(dataset):
    """Test SMILES strings are correctly associated."""
    print("\n=== Test 4: SMILES Correspondence ===")
    
    # Known SMILES from first few entries
    example_0 = dataset[0]
    example_1 = dataset[1]
    
    assert 'smiles' in example_0
    assert len(example_0['smiles']) > 0
    assert example_0['smiles'] != example_1['smiles']  # Should be different
    
    print(f"✓ SMILES correctly associated")
    print(f"  Example 0: {example_0['smiles']}")
    print(f"  Example 1: {example_1['smiles']}")


def print_example_details(example):
    """Print detailed information about an example."""
    print("\n=== Example Details ===")
    print(f"SMILES: {example['smiles']}")
    print(f"\nMessages:")
    for i, msg in enumerate(example['messages']):
        content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
        print(f"  [{i}] {msg['role']}: {content}")
    
    print(f"\nGraph Structure:")
    value = example['graph_data']['value']
    for key, tensor in value.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")


def main():
    print("Testing Generated Molecule SFT Dataset\n")
    
    try:
        # Run tests
        dataset = test_dataset_loading()
        example = test_data_format(dataset)
        test_multiple_examples(dataset)
        test_smiles_correspondence(dataset)
        
        # Print example details
        print_example_details(example)
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED")
        print("="*50)
        print("\nThe generated dataset is ready for training!")
        print("Use MultiModalSFTDataset with use_combined_parquet=True:")
        print("  dataset = MultiModalSFTDataset(")
        print("      dataset_path='data/molecule_sft_train.parquet',")
        print("      use_combined_parquet=True")
        print("  )")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

