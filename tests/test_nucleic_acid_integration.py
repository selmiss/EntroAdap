#!/usr/bin/env python3
"""
Test script for nucleic acid integration.

This script verifies that the nucleic acid embedder and dataloader
work correctly with DNA/RNA data.
"""

import sys
from pathlib import Path
import torch
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.models.components.aa_embedder import AAEmbedder
from src.data_loader.graph_batch_utils import merge_nucleic_acid_graphs


@pytest.fixture
def embedder():
    """Fixture to create an AAEmbedder instance for testing."""
    return AAEmbedder(
        hidden_dim=256,
        num_rbf=32,
        rbf_max=10.0,
        protein_residue_id_scale=1000.0,
        nucleic_acid_residue_id_scale=500.0
    )


def test_embedder_initialization(embedder):
    """Test that AAEmbedder initializes with nucleic acid support."""
    # Check that nucleic acid components exist
    assert hasattr(embedder, 'nacid_node_embed'), "Missing nacid_node_embed"
    assert hasattr(embedder, 'nacid_residue_proj'), "Missing nacid_residue_proj"
    assert hasattr(embedder, 'nacid_node_combine'), "Missing nacid_node_combine"
    assert hasattr(embedder, 'nacid_node_offset'), "Missing nacid_node_offset"
    
    # Check configuration
    assert embedder.hidden_dim == 256
    assert embedder.nucleic_acid_residue_id_scale == 500.0


def test_dna_embedding(embedder):
    """Test DNA graph embedding."""
    
    # Create synthetic DNA graph
    # Node features: [atomic_number(119), atom_name(30), nucleotide(11), chain(27), residue_id(cont), is_backbone(2), is_phosphate(2)]
    num_atoms = 100
    num_edges = 200
    
    node_feat = torch.zeros(num_atoms, 7)
    node_feat[:, 0] = torch.randint(0, 119, (num_atoms,))  # atomic_number
    node_feat[:, 1] = torch.randint(0, 30, (num_atoms,))   # atom_name
    node_feat[:, 2] = torch.randint(0, 11, (num_atoms,))   # nucleotide
    node_feat[:, 3] = torch.randint(0, 27, (num_atoms,))   # chain
    node_feat[:, 4] = torch.randint(0, 100, (num_atoms,))  # residue_id (continuous)
    node_feat[:, 5] = torch.randint(0, 2, (num_atoms,))    # is_backbone
    node_feat[:, 6] = torch.randint(0, 2, (num_atoms,))    # is_phosphate
    
    data = {
        'modality': 'dna',
        'value': {
            'node_feat': node_feat.float(),
            'edge_index': torch.randint(0, num_atoms, (2, num_edges)),
            'edge_attr': torch.rand(num_edges, 1),
            'pos': torch.randn(num_atoms, 3)
        }
    }
    
    # Embed the graph
    output = embedder(data)
    
    # Check output format
    assert 'node_emb' in output, "Missing node_emb in output"
    assert 'edge_emb' in output, "Missing edge_emb in output"
    assert 'edge_index' in output, "Missing edge_index in output"
    assert 'pos' in output, "Missing pos in output"
    
    # Check shapes
    assert output['node_emb'].shape == (num_atoms, embedder.hidden_dim), \
        f"Wrong node_emb shape: {output['node_emb'].shape}"
    assert output['edge_emb'].shape == (num_edges, embedder.hidden_dim), \
        f"Wrong edge_emb shape: {output['edge_emb'].shape}"
    assert output['edge_index'].shape == (2, num_edges), \
        f"Wrong edge_index shape: {output['edge_index'].shape}"
    assert output['pos'].shape == (num_atoms, 3), \
        f"Wrong pos shape: {output['pos'].shape}"


def test_rna_embedding(embedder):
    """Test RNA graph embedding."""
    
    # Create synthetic RNA graph
    num_atoms = 150
    num_edges = 300
    
    node_feat = torch.zeros(num_atoms, 7)
    node_feat[:, 0] = torch.randint(0, 119, (num_atoms,))  # atomic_number
    node_feat[:, 1] = torch.randint(0, 30, (num_atoms,))   # atom_name
    node_feat[:, 2] = torch.randint(0, 11, (num_atoms,))   # nucleotide
    node_feat[:, 3] = torch.randint(0, 27, (num_atoms,))   # chain
    node_feat[:, 4] = torch.randint(0, 100, (num_atoms,))  # residue_id (continuous)
    node_feat[:, 5] = torch.randint(0, 2, (num_atoms,))    # is_backbone
    node_feat[:, 6] = torch.randint(0, 2, (num_atoms,))    # is_phosphate
    
    data = {
        'modality': 'rna',
        'value': {
            'node_feat': node_feat.float(),
            'edge_index': torch.randint(0, num_atoms, (2, num_edges)),
            'edge_attr': torch.rand(num_edges, 1),
            'pos': torch.randn(num_atoms, 3)
        }
    }
    
    # Embed the graph
    output = embedder(data)
    
    # Check output format
    assert output['node_emb'].shape == (num_atoms, embedder.hidden_dim)
    assert output['edge_emb'].shape == (num_edges, embedder.hidden_dim)


def test_batch_merging():
    """Test nucleic acid graph batching."""
    
    # Create multiple synthetic graphs
    graphs = []
    for i in range(3):
        num_atoms = 50 + i * 10
        num_edges = 100 + i * 20
        
        node_feat = torch.zeros(num_atoms, 7)
        node_feat[:, 0] = torch.randint(0, 119, (num_atoms,))
        node_feat[:, 1] = torch.randint(0, 30, (num_atoms,))
        node_feat[:, 2] = torch.randint(0, 11, (num_atoms,))
        node_feat[:, 3] = torch.randint(0, 27, (num_atoms,))
        node_feat[:, 4] = torch.randint(0, 100, (num_atoms,))
        node_feat[:, 5] = torch.randint(0, 2, (num_atoms,))
        node_feat[:, 6] = torch.randint(0, 2, (num_atoms,))
        
        graph = {
            'node_feat': node_feat.float(),
            'edge_index': torch.randint(0, num_atoms, (2, num_edges)),
            'edge_attr': torch.rand(num_edges, 1),
            'pos': torch.randn(num_atoms, 3)
        }
        graphs.append(graph)
    
    # Merge graphs
    merged = merge_nucleic_acid_graphs(graphs)
    
    # Check merged format
    assert 'node_feat' in merged, "Missing node_feat in merged"
    assert 'edge_index' in merged, "Missing edge_index in merged"
    assert 'edge_attr' in merged, "Missing edge_attr in merged"
    assert 'pos' in merged, "Missing pos in merged"
    assert 'batch' in merged, "Missing batch in merged"
    
    # Check sizes
    total_atoms = sum(g['node_feat'].size(0) for g in graphs)
    total_edges = sum(g['edge_index'].size(1) for g in graphs)
    
    assert merged['node_feat'].size(0) == total_atoms, \
        f"Wrong total atoms: {merged['node_feat'].size(0)} vs {total_atoms}"
    assert merged['edge_index'].size(1) == total_edges, \
        f"Wrong total edges: {merged['edge_index'].size(1)} vs {total_edges}"
    assert merged['batch'].size(0) == total_atoms, \
        f"Wrong batch size: {merged['batch'].size(0)} vs {total_atoms}"
    
    # Check batch assignments
    assert merged['batch'].min() == 0, "Batch should start at 0"
    assert merged['batch'].max() == len(graphs) - 1, "Batch should end at num_graphs - 1"


def test_modality_routing(embedder):
    """Test that embedder correctly routes different modalities."""
    
    modalities = ['protein', 'molecule', 'dna', 'rna']
    
    for modality in modalities:
        num_atoms = 100
        num_edges = 200
        
        if modality == 'molecule':
            # Molecule has different structure
            # Molecule node features (9): [atomic_num(119), chirality(4), degree(12), charge(12), numH(10), radical(6), hybrid(6), aromatic(2), ring(2)]
            node_feat = torch.zeros(num_atoms, 9)
            node_feat[:, 0] = torch.randint(0, 119, (num_atoms,))  # atomic_num
            node_feat[:, 1] = torch.randint(0, 4, (num_atoms,))    # chirality
            node_feat[:, 2] = torch.randint(0, 12, (num_atoms,))   # degree
            node_feat[:, 3] = torch.randint(0, 12, (num_atoms,))   # charge
            node_feat[:, 4] = torch.randint(0, 10, (num_atoms,))   # numH
            node_feat[:, 5] = torch.randint(0, 6, (num_atoms,))    # radical
            node_feat[:, 6] = torch.randint(0, 6, (num_atoms,))    # hybrid
            node_feat[:, 7] = torch.randint(0, 2, (num_atoms,))    # aromatic
            node_feat[:, 8] = torch.randint(0, 2, (num_atoms,))    # ring
            
            data = {
                'modality': modality,
                'value': {
                    'node_feat': node_feat.float(),
                    'edge_index': torch.randint(0, num_atoms, (2, num_edges)),
                    'edge_feat_dist': torch.rand(num_edges, 1),
                    'pos': torch.randn(num_atoms, 3)
                }
            }
        else:
            # Protein, DNA, RNA have similar structure
            node_feat_dim = 7
            node_feat = torch.zeros(num_atoms, node_feat_dim)
            
            if modality == 'protein':
                # Protein: [atomic_number(119), atom_name(46), residue(24), chain(27), residue_id, is_backbone(2), is_ca(2)]
                node_feat[:, 0] = torch.randint(0, 119, (num_atoms,))
                node_feat[:, 1] = torch.randint(0, 46, (num_atoms,))
                node_feat[:, 2] = torch.randint(0, 24, (num_atoms,))
                node_feat[:, 3] = torch.randint(0, 27, (num_atoms,))
                node_feat[:, 4] = torch.randint(0, 100, (num_atoms,))
                node_feat[:, 5] = torch.randint(0, 2, (num_atoms,))
                node_feat[:, 6] = torch.randint(0, 2, (num_atoms,))
            else:  # DNA or RNA
                # Nucleic acid: [atomic_number(119), atom_name(30), nucleotide(11), chain(27), residue_id, is_backbone(2), is_phosphate(2)]
                node_feat[:, 0] = torch.randint(0, 119, (num_atoms,))
                node_feat[:, 1] = torch.randint(0, 30, (num_atoms,))
                node_feat[:, 2] = torch.randint(0, 11, (num_atoms,))
                node_feat[:, 3] = torch.randint(0, 27, (num_atoms,))
                node_feat[:, 4] = torch.randint(0, 100, (num_atoms,))
                node_feat[:, 5] = torch.randint(0, 2, (num_atoms,))
                node_feat[:, 6] = torch.randint(0, 2, (num_atoms,))
            
            data = {
                'modality': modality,
                'value': {
                    'node_feat': node_feat.float(),
                    'edge_index': torch.randint(0, num_atoms, (2, num_edges)),
                    'edge_attr': torch.rand(num_edges, 1),
                    'pos': torch.randn(num_atoms, 3)
                }
            }
        
        # Embed the graph
        output = embedder(data)
        
        # Check output
        assert output['node_emb'].shape == (num_atoms, embedder.hidden_dim)


def test_invalid_modality(embedder):
    """Test that invalid modality raises error."""
    data = {
        'modality': 'invalid_modality',
        'value': {
            'node_feat': torch.randn(10, 7),
            'edge_index': torch.randint(0, 10, (2, 20)),
            'edge_attr': torch.rand(20, 1),
            'pos': torch.randn(10, 3)
        }
    }
    
    with pytest.raises(ValueError, match="Unknown modality"):
        embedder(data)

