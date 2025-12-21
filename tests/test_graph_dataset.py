"""
Tests for Graph Dataset and DataLoader
"""

import pytest
import torch
import tempfile
import os
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from src.data_loader.graph_dataset import GraphDataset, GraphBatchCollator


def create_test_parquet_protein(path: str, num_samples: int = 10):
    """Create a test parquet file with protein graphs."""
    data = {
        'modality': [],
        'node_feat': [],
        'edge_index': [],
        'edge_attr': [],
        'pos': [],
    }
    
    for _ in range(num_samples):
        N = torch.randint(10, 20, (1,)).item()
        E = torch.randint(20, 40, (1,)).item()
        
        # Protein features: [atomic_num, atom_name, residue, chain, residue_id, is_backbone, is_ca]
        node_feat = torch.cat([
            torch.randint(0, 119, (N, 1)),
            torch.randint(0, 46, (N, 1)),
            torch.randint(0, 24, (N, 1)),
            torch.randint(0, 27, (N, 1)),
            torch.randint(1, 100, (N, 1)),
            torch.randint(0, 2, (N, 1)),
            torch.randint(0, 2, (N, 1)),
        ], dim=1).tolist()
        
        edge_index = torch.randint(0, N, (2, E)).tolist()
        edge_attr = (torch.rand(E) * 10).tolist()
        pos = torch.randn(N, 3).tolist()
        
        data['modality'].append('protein')
        data['node_feat'].append(node_feat)
        data['edge_index'].append(edge_index)
        data['edge_attr'].append(edge_attr)
        data['pos'].append(pos)
    
    table = pa.table(data)
    pq.write_table(table, path)


def create_test_parquet_molecule(path: str, num_samples: int = 10):
    """Create a test parquet file with molecule graphs."""
    data = {
        'modality': [],
        'node_feat': [],
        'edge_index': [],
        'edge_feat_dist': [],
        'chem_edge_index': [],
        'chem_edge_feat_cat': [],
        'pos': [],
    }
    
    for _ in range(num_samples):
        N = torch.randint(5, 15, (1,)).item()
        E_spatial = torch.randint(10, 25, (1,)).item()
        E_chem = torch.randint(5, 15, (1,)).item()
        
        # Molecule features: [atomic_num, chirality, degree, charge, numH, radical, hybrid, aromatic, ring]
        node_feat = torch.cat([
            torch.randint(0, 119, (N, 1)),
            torch.randint(0, 4, (N, 1)),
            torch.randint(0, 12, (N, 1)),
            torch.randint(0, 12, (N, 1)),
            torch.randint(0, 10, (N, 1)),
            torch.randint(0, 6, (N, 1)),
            torch.randint(0, 6, (N, 1)),
            torch.randint(0, 2, (N, 1)),
            torch.randint(0, 2, (N, 1)),
        ], dim=1).tolist()
        
        edge_index = torch.randint(0, N, (2, E_spatial)).tolist()
        edge_feat_dist = (torch.rand(E_spatial) * 10).tolist()
        chem_edge_index = torch.randint(0, N, (2, E_chem)).tolist()
        
        # Chemical edge features: [bond_type, stereo, conjugated]
        chem_edge_feat_cat = torch.cat([
            torch.randint(0, 5, (E_chem, 1)),
            torch.randint(0, 6, (E_chem, 1)),
            torch.randint(0, 2, (E_chem, 1)),
        ], dim=1).tolist()
        
        pos = torch.randn(N, 3).tolist()
        
        data['modality'].append('molecule')
        data['node_feat'].append(node_feat)
        data['edge_index'].append(edge_index)
        data['edge_feat_dist'].append(edge_feat_dist)
        data['chem_edge_index'].append(chem_edge_index)
        data['chem_edge_feat_cat'].append(chem_edge_feat_cat)
        data['pos'].append(pos)
    
    table = pa.table(data)
    pq.write_table(table, path)


@pytest.fixture
def protein_parquet_file():
    """Create temporary protein parquet file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "protein.parquet")
        create_test_parquet_protein(path, num_samples=20)
        yield path


@pytest.fixture
def molecule_parquet_file():
    """Create temporary molecule parquet file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "molecule.parquet")
        create_test_parquet_molecule(path, num_samples=20)
        yield path


@pytest.fixture
def collator():
    """Create graph batch collator."""
    return GraphBatchCollator(
        node_mask_prob=0.15,
        noise_std=0.1,
        num_dist_bins=32,
        dist_min=0.0,
        dist_max=20.0,
    )


class TestGraphDataset:
    """Tests for GraphDataset."""
    
    def test_protein_dataset_load(self, protein_parquet_file):
        """Test loading protein dataset."""
        dataset = GraphDataset(protein_parquet_file, split='train')
        assert len(dataset) == 20
    
    def test_protein_dataset_getitem(self, protein_parquet_file):
        """Test getting protein sample."""
        dataset = GraphDataset(protein_parquet_file, split='train')
        sample = dataset[0]
        
        assert 'modality' in sample
        assert sample['modality'] == 'protein'
        assert 'value' in sample
        
        value = sample['value']
        assert 'node_feat' in value
        assert 'edge_index' in value
        assert 'edge_attr' in value
        assert 'pos' in value
        
        # Check shapes
        N = value['node_feat'].size(0)
        E = value['edge_index'].size(1)
        assert value['node_feat'].shape == (N, 7)
        assert value['edge_index'].shape == (2, E)
        assert value['edge_attr'].shape[0] == E
        assert value['pos'].shape == (N, 3)
    
    def test_molecule_dataset_load(self, molecule_parquet_file):
        """Test loading molecule dataset."""
        dataset = GraphDataset(molecule_parquet_file, split='train')
        assert len(dataset) == 20
    
    def test_molecule_dataset_getitem(self, molecule_parquet_file):
        """Test getting molecule sample."""
        dataset = GraphDataset(molecule_parquet_file, split='train')
        sample = dataset[0]
        
        assert sample['modality'] == 'molecule'
        value = sample['value']
        
        assert 'node_feat' in value
        assert 'edge_index' in value
        assert 'edge_feat_dist' in value
        assert 'chem_edge_index' in value
        assert 'chem_edge_feat_cat' in value
        assert 'pos' in value
        
        # Check shapes
        N = value['node_feat'].size(0)
        assert value['node_feat'].shape == (N, 9)
        assert value['pos'].shape == (N, 3)
    
    def test_dataset_iteration(self, protein_parquet_file):
        """Test iterating through dataset."""
        dataset = GraphDataset(protein_parquet_file, split='train')
        count = 0
        for sample in dataset:
            assert 'modality' in sample
            count += 1
        assert count == 20


class TestGraphBatchCollator:
    """Tests for GraphBatchCollator."""
    
    def test_collator_initialization(self):
        """Test collator initialization."""
        collator = GraphBatchCollator(
            node_mask_prob=0.2,
            noise_std=0.05,
        )
        assert collator.node_mask_prob == 0.2
        assert collator.noise_std == 0.05
    
    def test_protein_batch_collation(self, protein_parquet_file, collator):
        """Test collating protein graphs."""
        dataset = GraphDataset(protein_parquet_file, split='train')
        batch_list = [dataset[i] for i in range(4)]
        
        batch = collator(batch_list)
        
        # Check batch structure
        assert 'data' in batch
        assert 'batch' in batch
        assert 'node_mask' in batch
        assert 'edge_mask' in batch
        assert 'element_labels' in batch
        assert 'dist_labels' in batch
        assert 'noise_labels' in batch
        
        # Check data structure
        data = batch['data']
        assert data['modality'] == 'protein'
        assert 'value' in data
        
        value = data['value']
        assert 'node_feat' in value
        assert 'edge_index' in value
        assert 'pos' in value
        assert 'batch' in value
        
        # Check batch assignment
        num_nodes = value['node_feat'].size(0)
        assert batch['batch'].shape == (num_nodes,)
        assert batch['batch'].max().item() == 3  # 4 graphs (0-3)
        
        # Check masks
        assert batch['node_mask'].dtype == torch.bool
        assert batch['edge_mask'].dtype == torch.bool
        assert batch['node_mask'].shape == (num_nodes,)
        
        # Check labels
        assert batch['element_labels'].shape == (num_nodes,)
        assert batch['noise_labels'].shape == (num_nodes, 3)
    
    def test_molecule_batch_collation(self, molecule_parquet_file, collator):
        """Test collating molecule graphs."""
        dataset = GraphDataset(molecule_parquet_file, split='train')
        batch_list = [dataset[i] for i in range(4)]
        
        batch = collator(batch_list)
        
        data = batch['data']
        assert data['modality'] == 'molecule'
        
        value = data['value']
        num_nodes = value['node_feat'].size(0)
        
        # Check batch structure
        assert batch['batch'].shape == (num_nodes,)
        assert batch['node_mask'].shape == (num_nodes,)
        assert batch['element_labels'].shape == (num_nodes,)
    
    def test_masking_applied(self, protein_parquet_file, collator):
        """Test that masking is applied correctly."""
        dataset = GraphDataset(protein_parquet_file, split='train')
        batch_list = [dataset[i] for i in range(4)]
        
        batch = collator(batch_list)
        
        # Check some nodes are masked
        num_masked = batch['node_mask'].sum().item()
        num_nodes = batch['node_mask'].shape[0]
        
        # With 0.15 probability, expect roughly 15% masked (but allow variance)
        assert 0 < num_masked < num_nodes
        
        # Check some edges are masked
        num_masked_edges = batch['edge_mask'].sum().item()
        num_edges = batch['edge_mask'].shape[0]
        assert 0 < num_masked_edges < num_edges
    
    def test_coordinate_noise(self, protein_parquet_file):
        """Test coordinate noise is applied to masked nodes."""
        collator = GraphBatchCollator(noise_std=0.5)
        dataset = GraphDataset(protein_parquet_file, split='train')
        batch_list = [dataset[i] for i in range(2)]
        
        batch = collator(batch_list)
        
        # Noise labels should have non-zero values
        noise_labels = batch['noise_labels']
        assert noise_labels.abs().sum() > 0
        
        # Positions should be modified (noisy)
        pos = batch['data']['value']['pos']
        assert pos.shape[-1] == 3
    
    def test_soft_distance_targets(self, protein_parquet_file):
        """Test soft distance target generation."""
        collator = GraphBatchCollator(
            use_soft_dist_targets=True,
            soft_dist_sigma=0.5,
            num_dist_bins=32,
        )
        dataset = GraphDataset(protein_parquet_file, split='train')
        batch_list = [dataset[i] for i in range(2)]
        
        batch = collator(batch_list)
        
        dist_labels = batch['dist_labels']
        
        # Should be 2D (soft targets)
        assert dist_labels.dim() == 2
        assert dist_labels.shape[1] == 32
        
        # Should sum to 1 (probability distribution)
        sums = dist_labels.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_hard_distance_targets(self, protein_parquet_file):
        """Test hard distance targets."""
        collator = GraphBatchCollator(use_soft_dist_targets=False)
        dataset = GraphDataset(protein_parquet_file, split='train')
        batch_list = [dataset[i] for i in range(2)]
        
        batch = collator(batch_list)
        
        dist_labels = batch['dist_labels']
        
        # Should be 1D (hard targets)
        assert dist_labels.dim() == 1
        assert (dist_labels >= 0).all()
    
    def test_element_labels_preserved(self, protein_parquet_file, collator):
        """Test element labels are correctly extracted."""
        dataset = GraphDataset(protein_parquet_file, split='train')
        sample = dataset[0]
        
        # Get original element IDs (first column)
        original_elements = sample['value']['node_feat'][:, 0].long()
        
        batch = collator([sample])
        
        # Check element labels match original
        assert torch.equal(batch['element_labels'], original_elements)
    
    def test_mixed_modality_error(self, protein_parquet_file, molecule_parquet_file, collator):
        """Test error when mixing modalities."""
        protein_dataset = GraphDataset(protein_parquet_file, split='train')
        molecule_dataset = GraphDataset(molecule_parquet_file, split='train')
        
        batch_list = [protein_dataset[0], molecule_dataset[0]]
        
        with pytest.raises(ValueError, match="same modality"):
            collator(batch_list)
    
    def test_single_graph_batch(self, protein_parquet_file, collator):
        """Test batching single graph."""
        dataset = GraphDataset(protein_parquet_file, split='train')
        batch = collator([dataset[0]])
        
        # Should still work
        assert 'data' in batch
        assert batch['batch'].max().item() == 0  # Only one graph
    
    def test_large_batch(self, protein_parquet_file, collator):
        """Test batching many graphs."""
        dataset = GraphDataset(protein_parquet_file, split='train')
        batch_list = [dataset[i] for i in range(10)]
        
        batch = collator(batch_list)
        
        assert batch['batch'].max().item() == 9  # 10 graphs (0-9)
    
    def test_batch_indices_correct(self, protein_parquet_file, collator):
        """Test batch indices are assigned correctly."""
        dataset = GraphDataset(protein_parquet_file, split='train')
        batch_list = [dataset[i] for i in range(3)]
        
        batch = collator(batch_list)
        batch_indices = batch['batch']
        
        # Count nodes per graph
        for i in range(3):
            count = (batch_indices == i).sum().item()
            assert count > 0  # Each graph should have some nodes


class TestIntegration:
    """Integration tests with trainer."""
    
    def test_batch_format_for_trainer(self, protein_parquet_file, collator):
        """Test batch format is compatible with trainer."""
        from src.models.geo_encoder import GeoEncoder
        from src.trainer.masked_reconstruction import MaskedReconstructionTrainer
        
        # Create model
        encoder = GeoEncoder(hidden_dim=64, num_layers=2)
        trainer = MaskedReconstructionTrainer(encoder=encoder)
        
        # Create batch
        dataset = GraphDataset(protein_parquet_file, split='train')
        batch_list = [dataset[i] for i in range(2)]
        batch = collator(batch_list)
        
        # Forward pass
        result = trainer(
            data=batch['data'],
            batch=batch['batch'],
            node_mask=batch['node_mask'],
            edge_mask=batch['edge_mask'],
            element_labels=batch['element_labels'],
            dist_labels=batch['dist_labels'],
            noise_labels=batch['noise_labels'],
            compute_loss=True,
        )
        
        # Check output
        assert 'loss' in result
        assert result['loss'].ndim == 0  # Scalar loss
        assert 'element_loss' in result
        assert 'dist_loss' in result
        assert 'noise_loss' in result
    
    def test_dataloader_iteration(self, protein_parquet_file, collator):
        """Test iterating through dataloader."""
        from torch.utils.data import DataLoader
        
        dataset = GraphDataset(protein_parquet_file, split='train')
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collator,
            shuffle=False,
        )
        
        batches = list(dataloader)
        assert len(batches) == 5  # 20 samples / 4 = 5 batches
        
        for batch in batches:
            assert 'data' in batch
            assert 'batch' in batch
            assert 'node_mask' in batch

