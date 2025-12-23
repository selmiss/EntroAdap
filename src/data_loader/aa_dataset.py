"""
Dataset and DataLoader for Masked Reconstruction Training

This module provides:
1. GraphDataset: Loads graph data from parquet files
2. GraphBatchCollator: Batches graphs using PyG and applies masking
3. Helper functions for data loading
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, Any, List, Optional

from .graph_batch_utils import merge_protein_graphs, merge_molecule_graphs, bfs_patch_masking


class GraphDataset(Dataset):
    """
    Dataset for loading graph structures from HuggingFace parquet format.
    
    Expected parquet schema:
        - modality: str ('protein' or 'molecule')
        - node_feat: List[List[int/float]] - node features
        - edge_index: List[List[int]] - [2, E] edge connectivity
        - pos: List[List[float]] - [N, 3] coordinates
        - edge_attr: List[float] - edge attributes (for protein)
        - chem_edge_index: List[List[int]] - chemical edges (for molecule)
        - chem_edge_feat_cat: List[List[int]] - chemical edge features (for molecule)
        - edge_feat_dist: List[float] - spatial edge distances (for molecule)
    
    Args:
        dataset_path: Path to parquet file or HF dataset
        split: Dataset split ('train', 'validation', 'test')
        cache_dir: Optional cache directory for HF datasets
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = 'train',
        cache_dir: Optional[str] = None,
    ):
        # When loading from a single parquet file, HF datasets exposes it as 'train' split
        # To avoid "Unknown split" errors, we load without split specification first
        dataset = load_dataset(
            'parquet',
            data_files=dataset_path,
            cache_dir=cache_dir,
        )
        # The dataset will be a DatasetDict; extract the actual split
        # Single file -> exposed as 'train' split
        if 'train' in dataset:
            self.dataset = dataset['train']
        else:
            # If somehow structured differently, take the first available split
            self.dataset = dataset[list(dataset.keys())[0]]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single graph sample.
        
        Returns:
            Dictionary with 'modality' and 'value' keys matching AAEmbedder input format
        """
        item = self.dataset[idx]
        modality = item['modality']
        
        # Convert lists to tensors
        value = {}
        
        # Common fields
        value['node_feat'] = torch.tensor(item['node_feat'], dtype=torch.float32)
        value['pos'] = torch.tensor(item['pos'], dtype=torch.float32)
        value['edge_index'] = torch.tensor(item['edge_index'], dtype=torch.long)
        
        # Modality-specific fields
        if modality == 'protein':
            if 'edge_attr' in item:
                value['edge_attr'] = torch.tensor(item['edge_attr'], dtype=torch.float32)
                if value['edge_attr'].dim() == 1:
                    value['edge_attr'] = value['edge_attr'].unsqueeze(-1)
        
        elif modality == 'molecule':
            # Chemical edges
            if 'chem_edge_index' in item:
                value['chem_edge_index'] = torch.tensor(item['chem_edge_index'], dtype=torch.long)
            if 'chem_edge_feat_cat' in item:
                value['chem_edge_feat_cat'] = torch.tensor(item['chem_edge_feat_cat'], dtype=torch.long)
            
            # Spatial edges
            if 'edge_feat_dist' in item:
                value['edge_feat_dist'] = torch.tensor(item['edge_feat_dist'], dtype=torch.float32)
                if value['edge_feat_dist'].dim() == 1:
                    value['edge_feat_dist'] = value['edge_feat_dist'].unsqueeze(-1)
        
        return {
            'modality': modality,
            'value': value,
        }


class GraphBatchCollator:
    """
    Collates graph samples into PyG batches with masking for reconstruction.
    
    Applies masking strategy:
    - Node masking: Random subset of nodes for element prediction
    - Edge masking: Edges are masked when both endpoints are masked nodes
    - Coordinate noise: Gaussian noise added to masked node positions
    
    Args:
        node_mask_prob: Probability of masking each node
        noise_std: Standard deviation of coordinate noise
        num_dist_bins: Number of bins for distance discretization
        dist_min: Minimum distance for binning
        dist_max: Maximum distance for binning
        use_soft_dist_targets: Use soft distance targets (Gaussian smoothing)
        soft_dist_sigma: Sigma for Gaussian smoothing of distance targets
    """
    
    def __init__(
        self,
        node_mask_prob: float = 0.15,
        noise_std: float = 0.1,
        num_dist_bins: int = 64,
        dist_min: float = 0.0,
        dist_max: float = 20.0,
        use_soft_dist_targets: bool = False,
        soft_dist_sigma: float = 0.5,
    ):
        self.node_mask_prob = node_mask_prob
        self.noise_std = noise_std
        self.num_dist_bins = num_dist_bins
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.use_soft_dist_targets = use_soft_dist_targets
        self.soft_dist_sigma = soft_dist_sigma
        
        # Precompute bin edges
        self.dist_bins = torch.linspace(dist_min, dist_max, num_dist_bins + 1)
    
    def _apply_patch_masking(
        self,
        batch_data: Dict[str, Any],
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply BFS-based patch masking to nodes and edges.
        
        Returns:
            node_mask: Boolean mask [N]
            edge_mask: Boolean mask [E]
            element_labels: Ground truth element IDs [N]
            noise_labels: Ground truth coordinate noise [N, 3]
            dist_labels: Ground truth edge distances [E] or [E, num_bins] if soft targets
        """
        edge_index = batch_data['value']['edge_index']
        
        # Use BFS to create masked patches
        masked_node_set = bfs_patch_masking(
            edge_index=edge_index,
            num_nodes=num_nodes,
            target_mask_ratio=self.node_mask_prob,
            min_patch_size=1,
            max_patch_frac=0.02,
            accept_p=0.7,
            force_fill_to_target=True,
            make_undirected=True,
        )
        
        # Create node mask tensor
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for node_idx in masked_node_set:
            node_mask[node_idx] = True
        
        # Store original element IDs (first column of node_feat)
        element_labels = batch_data['value']['node_feat'][:, 0].long()
        
        # Get edge distances from edge attributes (BEFORE adding noise to positions)
        if 'edge_attr' in batch_data['value']:
            edge_distances = batch_data['value']['edge_attr'].squeeze(-1)
        elif 'edge_feat_dist' in batch_data['value']:
            edge_distances = batch_data['value']['edge_feat_dist'].squeeze(-1)
        else:
            # Fallback: compute from original positions
            src_pos = batch_data['value']['pos'][edge_index[0]]
            dst_pos = batch_data['value']['pos'][edge_index[1]]
            edge_distances = torch.norm(dst_pos - src_pos, dim=-1)
        
        # Compute distance labels
        if self.use_soft_dist_targets:
            dist_labels = self._distances_to_soft_targets(edge_distances)
        else:
            dist_labels = edge_distances
        
        # Generate coordinate noise
        noise = torch.randn(num_nodes, 3) * self.noise_std
        noise_labels = noise.clone()
        
        # Apply noise to masked nodes
        batch_data['value']['pos'] = batch_data['value']['pos'] + noise * node_mask.unsqueeze(-1).float()
        
        # Mask node features (set first column to 0 for masked nodes)
        batch_data['value']['node_feat'][node_mask, 0] = 0
        
        # Mask edges within the masked patch
        # An edge is masked if BOTH endpoints are masked
        src_masked = node_mask[edge_index[0]]
        dst_masked = node_mask[edge_index[1]]
        edge_mask = src_masked & dst_masked
        
        # Zero out masked edge features
        if 'edge_attr' in batch_data['value']:
            batch_data['value']['edge_attr'][edge_mask] = 0
        if 'edge_feat_dist' in batch_data['value']:
            batch_data['value']['edge_feat_dist'][edge_mask] = 0
        
        return node_mask, edge_mask, element_labels, noise_labels, dist_labels
    
    def _distances_to_soft_targets(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to soft probability distributions over bins.
        
        Args:
            distances: Ground truth distances [E]
        
        Returns:
            Soft targets [E, num_dist_bins]
        """
        # Clamp distances to valid range
        distances = torch.clamp(distances, self.dist_min, self.dist_max)
        
        bin_centers = (self.dist_bins[:-1] + self.dist_bins[1:]) / 2
        diff = distances.unsqueeze(-1) - bin_centers.unsqueeze(0)  # [E, num_bins]
        soft_targets = torch.exp(-diff ** 2 / (2 * self.soft_dist_sigma ** 2))
        soft_targets = soft_targets / (soft_targets.sum(dim=-1, keepdim=True) + 1e-12)
        return soft_targets
    
    def _collate_graphs(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate graphs using PyG's batching mechanism.
        
        Returns standardized batch dictionary with 'modality' and 'value' keys.
        """
        # Group by modality
        modalities = [item['modality'] for item in batch_list]
        
        # For simplicity, we require all graphs in batch to have same modality
        # (This is typical for training - separate dataloaders for protein/molecule)
        if len(set(modalities)) > 1:
            raise ValueError("All graphs in batch must have the same modality")
        
        modality = modalities[0]
        
        # Extract value dicts
        graphs = [item['value'] for item in batch_list]
        
        # Merge graphs based on modality
        if modality == 'protein':
            merged = merge_protein_graphs(graphs)
        else:
            merged = merge_molecule_graphs(graphs)
        
        return {
            'modality': modality,
            'value': merged,
        }
    
    def __call__(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of graphs with masking.
        
        Args:
            batch_list: List of graph dictionaries from dataset
        
        Returns:
            Dictionary with:
                - data: Graph batch with 'modality' and 'value' keys
                - batch: Batch assignment tensor [N]
                - node_mask: Boolean mask for masked nodes [N]
                - edge_mask: Boolean mask for masked edges [E]
                - element_labels: Ground truth element IDs [N]
                - dist_labels: Ground truth distances [E] or [E, num_bins]
                - noise_labels: Ground truth coordinate noise [N, 3]
        """
        # First collate graphs into a single batch
        batch_data = self._collate_graphs(batch_list)
        
        num_nodes = batch_data['value']['node_feat'].size(0)
        
        # Apply BFS-based patch masking (handles both nodes and edges)
        node_mask, edge_mask, element_labels, noise_labels, dist_labels = self._apply_patch_masking(
            batch_data, num_nodes
        )
        
        return {
            'data': batch_data,
            'batch': batch_data['value']['batch'],
            'node_mask': node_mask,
            'edge_mask': edge_mask,
            'element_labels': element_labels,
            'dist_labels': dist_labels,
            'noise_labels': noise_labels,
        }

