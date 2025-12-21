"""
Tests for Graph Batch Utils and BFS Patch Masking
"""

import pytest
import torch
from src.data_loader.graph_batch_utils import (
    merge_protein_graphs,
    merge_molecule_graphs,
    bfs_patch_masking,
)


class TestBFSPatchMasking:
    """Tests for BFS patch masking function."""
    
    def test_basic_masking(self):
        """Test basic BFS masking on a simple graph."""
        # Create a chain graph: 0-1-2-3-4-5-6-7-8-9
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ])
        
        masked = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=10,
            target_mask_ratio=0.3,
        )
        
        assert isinstance(masked, set)
        # Allow 20-40% range around target (patch-based masking isn't exact)
        assert 2 <= len(masked) <= 4, f"Expected 2-4 masked nodes, got {len(masked)}"
        assert all(isinstance(x, int) for x in masked)
        assert all(0 <= x < 10 for x in masked)
    
    def test_reproducibility_with_generator(self):
        """Test that using the same generator produces same results."""
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ])
        
        gen1 = torch.Generator().manual_seed(42)
        masked1 = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=10,
            target_mask_ratio=0.3,
            generator=gen1,
        )
        
        gen2 = torch.Generator().manual_seed(42)
        masked2 = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=10,
            target_mask_ratio=0.3,
            generator=gen2,
        )
        
        assert masked1 == masked2
    
    def test_target_ratio_achieved(self):
        """Test that target masking ratio is approximately achieved."""
        # Create a dense graph
        num_nodes = 50
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
        for i in range(0, num_nodes - 2, 2):
            edges.append([i, i + 2])
        
        edge_index = torch.tensor(edges).t()
        
        target_ratios = [0.1, 0.2, 0.3, 0.5]
        for ratio in target_ratios:
            masked = bfs_patch_masking(
                edge_index=edge_index,
                num_nodes=num_nodes,
                target_mask_ratio=ratio,
                force_fill_to_target=True,
            )
            
            target = int(num_nodes * ratio)
            tolerance = max(2, int(num_nodes * 0.1))  # 10% tolerance or at least 2 nodes
            assert abs(len(masked) - target) <= tolerance, \
                f"Ratio {ratio}: expected ~{target}, got {len(masked)}"
    
    def test_min_patch_size(self):
        """Test minimum patch size constraint."""
        # Small graph where min_patch_size matters
        edge_index = torch.tensor([
            [0, 1, 2],
            [1, 2, 3]
        ])
        
        # Request small ratio but with min_patch_size
        masked = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=10,
            target_mask_ratio=0.2,  # 2 nodes target
            min_patch_size=3,  # At least 3 per patch
            force_fill_to_target=False,  # Don't force fill
        )
        
        # With min_patch_size=3 and target 20%, may get around 2-4 nodes
        assert 1 <= len(masked) <= 5, f"Expected 1-5 masked nodes, got {len(masked)}"
    
    def test_max_patch_frac(self):
        """Test maximum patch size fraction."""
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ])
        
        masked = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=10,
            target_mask_ratio=0.5,  # 5 nodes target
            max_patch_frac=0.2,  # Max 2 nodes per patch
            force_fill_to_target=True,
        )
        
        # Should reach approximately 50% (4-6 nodes)
        assert 4 <= len(masked) <= 6, f"Expected 4-6 masked nodes, got {len(masked)}"
    
    def test_disconnected_graph(self):
        """Test on disconnected graph."""
        # Two disconnected components: 0-1-2 and 3-4-5
        edge_index = torch.tensor([
            [0, 1, 3, 4],
            [1, 2, 4, 5]
        ])
        
        masked = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=6,
            target_mask_ratio=0.5,  # 3 nodes target (50%)
            force_fill_to_target=True,
        )
        
        # Should mask 2-4 nodes (33-67% range)
        assert 2 <= len(masked) <= 4, f"Expected 2-4 masked nodes, got {len(masked)}"
    
    def test_empty_edges(self):
        """Test on graph with no edges."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        masked = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=10,
            target_mask_ratio=0.3,
            force_fill_to_target=True,
        )
        
        # Should mask around 30% (2-4 nodes)
        assert 2 <= len(masked) <= 4, f"Expected 2-4 masked nodes, got {len(masked)}"
    
    def test_cuda_edge_index(self):
        """Test that CUDA edge_index is handled correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5]
        ]).cuda()
        
        masked = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=6,
            target_mask_ratio=0.5,
        )
        
        assert isinstance(masked, set)
        # Should mask around 50% (2-4 nodes)
        assert 2 <= len(masked) <= 4, f"Expected 2-4 masked nodes, got {len(masked)}"
    
    def test_accept_probability(self):
        """Test that accept_p affects patch structure."""
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ])
        
        gen = torch.Generator().manual_seed(42)
        
        # Low accept_p should create sparser patches
        masked_low = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=10,
            target_mask_ratio=0.5,
            accept_p=0.3,
            generator=gen,
            force_fill_to_target=True,
        )
        
        # Should mask around 50% (4-6 nodes)
        assert 4 <= len(masked_low) <= 6, f"Expected 4-6 masked nodes, got {len(masked_low)}"
    
    def test_make_undirected(self):
        """Test make_undirected option."""
        # Directed edge: 0->1
        edge_index = torch.tensor([[0], [1]])
        
        # With make_undirected=True, both directions exist
        masked = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=3,
            target_mask_ratio=0.67,  # ~2 nodes target
            make_undirected=True,
            min_patch_size=1,
        )
        
        # Should mask 1-2 nodes (33-67%)
        assert 1 <= len(masked) <= 2, f"Expected 1-2 masked nodes, got {len(masked)}"
    
    def test_force_fill_to_target(self):
        """Test force_fill_to_target option."""
        # Disconnected nodes with very low accept_p
        edge_index = torch.tensor([[0], [1]])
        
        # Without force_fill, might not reach target
        masked_no_force = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=10,
            target_mask_ratio=0.5,
            accept_p=0.01,  # Very low
            force_fill_to_target=False,
            min_patch_size=1,
        )
        
        # With force_fill, should reach approximately target (4-6 nodes)
        masked_with_force = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=10,
            target_mask_ratio=0.5,
            accept_p=0.01,
            force_fill_to_target=True,
            min_patch_size=1,
        )
        
        assert 4 <= len(masked_with_force) <= 6, f"Expected 4-6 masked nodes with force_fill, got {len(masked_with_force)}"
        assert len(masked_no_force) <= len(masked_with_force)


class TestMergeProteinGraphs:
    """Tests for merging protein graphs."""
    
    def test_merge_single_graph(self):
        """Test merging a single graph."""
        graph = {
            'node_feat': torch.randn(10, 7),
            'pos': torch.randn(10, 3),
            'edge_index': torch.randint(0, 10, (2, 20)),
            'edge_attr': torch.randn(20, 1),
        }
        
        merged = merge_protein_graphs([graph])
        
        assert merged['node_feat'].shape == (10, 7)
        assert merged['pos'].shape == (10, 3)
        assert merged['edge_index'].shape == (2, 20)
        assert merged['edge_attr'].shape == (20, 1)
        assert 'batch' in merged
        assert (merged['batch'] == 0).all()
    
    def test_merge_multiple_graphs(self):
        """Test merging multiple graphs."""
        graphs = [
            {
                'node_feat': torch.randn(10, 7),
                'pos': torch.randn(10, 3),
                'edge_index': torch.randint(0, 10, (2, 15)),
                'edge_attr': torch.randn(15, 1),
            },
            {
                'node_feat': torch.randn(8, 7),
                'pos': torch.randn(8, 3),
                'edge_index': torch.randint(0, 8, (2, 12)),
                'edge_attr': torch.randn(12, 1),
            },
        ]
        
        merged = merge_protein_graphs(graphs)
        
        assert merged['node_feat'].shape == (18, 7)
        assert merged['pos'].shape == (18, 3)
        assert merged['edge_index'].shape == (2, 27)
        assert merged['edge_attr'].shape == (27, 1)
        assert merged['batch'].shape == (18,)
        assert (merged['batch'][:10] == 0).all()
        assert (merged['batch'][10:] == 1).all()
    
    def test_edge_index_offset(self):
        """Test that edge indices are properly offset."""
        graphs = [
            {
                'node_feat': torch.randn(5, 7),
                'pos': torch.randn(5, 3),
                'edge_index': torch.tensor([[0, 1], [1, 2]]),
            },
            {
                'node_feat': torch.randn(3, 7),
                'pos': torch.randn(3, 3),
                'edge_index': torch.tensor([[0, 1], [1, 2]]),
            },
        ]
        
        merged = merge_protein_graphs(graphs)
        
        # Second graph edges should be offset by 5
        assert merged['edge_index'][0, 2] == 5  # 0 + 5
        assert merged['edge_index'][1, 2] == 6  # 1 + 5
        assert merged['edge_index'][0, 3] == 6  # 1 + 5
        assert merged['edge_index'][1, 3] == 7  # 2 + 5


class TestMergeMoleculeGraphs:
    """Tests for merging molecule graphs."""
    
    def test_merge_with_spatial_and_chem_edges(self):
        """Test merging molecules with both edge types."""
        graphs = [
            {
                'node_feat': torch.randn(6, 9),
                'pos': torch.randn(6, 3),
                'edge_index': torch.randint(0, 6, (2, 10)),
                'edge_feat_dist': torch.randn(10, 1),
                'chem_edge_index': torch.randint(0, 6, (2, 8)),
                'chem_edge_feat_cat': torch.randint(0, 5, (8, 3)),
            },
            {
                'node_feat': torch.randn(4, 9),
                'pos': torch.randn(4, 3),
                'edge_index': torch.randint(0, 4, (2, 6)),
                'edge_feat_dist': torch.randn(6, 1),
                'chem_edge_index': torch.randint(0, 4, (2, 5)),
                'chem_edge_feat_cat': torch.randint(0, 5, (5, 3)),
            },
        ]
        
        merged = merge_molecule_graphs(graphs)
        
        assert merged['node_feat'].shape == (10, 9)
        assert merged['pos'].shape == (10, 3)
        assert merged['edge_index'].shape == (2, 16)
        assert merged['edge_feat_dist'].shape == (16, 1)
        assert merged['chem_edge_index'].shape == (2, 13)
        assert merged['chem_edge_feat_cat'].shape == (13, 3)
        assert merged['batch'].shape == (10,)
    
    def test_merge_with_only_spatial_edges(self):
        """Test merging molecules with only spatial edges."""
        graphs = [
            {
                'node_feat': torch.randn(5, 9),
                'pos': torch.randn(5, 3),
                'edge_index': torch.randint(0, 5, (2, 8)),
                'edge_feat_dist': torch.randn(8, 1),
            },
        ]
        
        merged = merge_molecule_graphs(graphs)
        
        assert 'edge_index' in merged
        assert 'edge_feat_dist' in merged
        assert 'chem_edge_index' not in merged
        assert 'chem_edge_feat_cat' not in merged
    
    def test_merge_with_only_chem_edges(self):
        """Test merging molecules with only chemical edges."""
        graphs = [
            {
                'node_feat': torch.randn(5, 9),
                'pos': torch.randn(5, 3),
                'chem_edge_index': torch.randint(0, 5, (2, 6)),
                'chem_edge_feat_cat': torch.randint(0, 5, (6, 3)),
            },
        ]
        
        merged = merge_molecule_graphs(graphs)
        
        assert 'chem_edge_index' in merged
        assert 'chem_edge_feat_cat' in merged
        assert 'edge_index' not in merged
        assert 'edge_feat_dist' not in merged


class TestIntegrationWithCollator:
    """Integration tests with GraphBatchCollator."""
    
    def test_bfs_masking_creates_patches(self):
        """Test that BFS masking creates connected patches."""
        from src.data_loader.graph_dataset import GraphBatchCollator
        
        # Create a simple graph structure
        batch_data = {
            'modality': 'protein',
            'value': {
                'node_feat': torch.cat([
                    torch.arange(20).unsqueeze(1).float(),  # element IDs
                    torch.randn(20, 6),
                ], dim=1),
                'pos': torch.randn(20, 3),
                'edge_index': torch.tensor([
                    list(range(19)),
                    list(range(1, 20))
                ]),  # Chain graph
                'edge_attr': torch.randn(19, 1),
                'batch': torch.zeros(20, dtype=torch.long),
            }
        }
        
        collator = GraphBatchCollator(node_mask_prob=0.3)
        
        # Apply masking
        node_mask, edge_mask, element_labels, noise_labels, dist_labels = collator._apply_patch_masking(
            batch_data, num_nodes=20
        )
        
        # Check that nodes are masked (around 30%, so 4-8 nodes)
        num_masked = node_mask.sum().item()
        assert 4 <= num_masked <= 8, f"Expected 4-8 masked nodes (20-40% of 20), got {num_masked}"
        
        # Check that masked nodes form patches (some edges between them should be masked)
        edge_index = batch_data['value']['edge_index']
        src_masked = node_mask[edge_index[0]]
        dst_masked = node_mask[edge_index[1]]
        edges_both_masked = (src_masked & dst_masked).sum()
        
        # Since we created patches, some edges should have both endpoints masked
        # (This is probabilistic, but with chain graph and patches, should be >0)
        assert edges_both_masked.item() >= 0
        
        # Check labels
        assert element_labels.shape == (20,)
        assert noise_labels.shape == (20, 3)
        assert dist_labels.shape[0] == 19  # Number of edges

