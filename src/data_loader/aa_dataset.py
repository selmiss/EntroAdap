"""
Dataset and DataLoader for Masked Reconstruction Training

This module provides:
1. GraphDataset: Loads graph data from parquet files
2. GraphBatchCollator: Batches graphs using PyG and applies masking
3. Helper functions for data loading
"""

import torch
from torch.utils.data import Dataset, Sampler
from datasets import load_dataset, concatenate_datasets
from typing import Dict, Any, List, Optional, Union
import random

from .graph_batch_utils import merge_protein_graphs, merge_molecule_graphs, merge_nucleic_acid_graphs, bfs_patch_masking


class GraphDataset(Dataset):
    """
    Dataset for loading graph structures from HuggingFace parquet format.
    
    Supports loading single or multiple parquet files. When multiple files are provided,
    they are concatenated into a single dataset, allowing mixed-modality training.
    
    Expected parquet schema:
        - modality: str ('protein', 'molecule', 'dna', or 'rna')
        - node_feat: List[List[int/float]] - node features
        - edge_index: List[List[int]] - [2, E] edge connectivity
        - pos: List[List[float]] - [N, 3] coordinates
        - edge_feat_dist: List[float] - spatial edge distances (unified format for all modalities)
        - edge_attr: List[float] - edge attributes (legacy format, supported for backward compatibility)
        - chem_edge_index: List[List[int]] - chemical edges (for molecule)
        - chem_edge_feat_cat: List[List[int]] - chemical edge features (for molecule)
    
    Args:
        dataset_path: Path(s) to parquet file(s). Can be:
            - Single string path
            - List of paths
            - Comma-separated string paths
        split: Dataset split ('train', 'validation', 'test')
        cache_dir: Optional cache directory for HF datasets
        max_samples_per_dataset: Optional list of max samples for each dataset.
            If provided, must match the number of datasets. None means no limit.
        stratified_val_ratio: If provided, will create a stratified validation split
            where each dataset contributes proportionally. Only used when creating val splits.
        max_atoms: Optional maximum number of atoms per structure. Structures exceeding
            this limit will be skipped during loading. None means no limit.
        max_edges: Optional maximum number of edges per structure. Structures exceeding
            this limit will be skipped during loading. None means no limit.
        skip_on_error: If True, skip samples that fail to load or exceed thresholds.
            If False, raise exceptions. Default: True.
    """
    
    def __init__(
        self,
        dataset_path: Union[str, List[str]],
        split: str = 'train',
        cache_dir: Optional[str] = None,
        max_samples_per_dataset: Optional[Union[int, List[Optional[int]]]] = None,
        stratified_val_ratio: Optional[float] = None,
        max_atoms: Optional[int] = None,
        max_edges: Optional[int] = None,
        skip_on_error: bool = True,
    ):
        # Store runtime filtering parameters
        self.max_atoms = max_atoms
        self.max_edges = max_edges
        self.skip_on_error = skip_on_error
        self._filtered_count = 0
        self._error_count = 0
        
        # Handle different input formats
        if isinstance(dataset_path, str):
            # Check if comma-separated paths
            if ',' in dataset_path:
                dataset_paths = [p.strip() for p in dataset_path.split(',')]
            else:
                dataset_paths = [dataset_path]
        else:
            dataset_paths = dataset_path
        
        # Handle max_samples_per_dataset
        if max_samples_per_dataset is None:
            max_samples_list = [None] * len(dataset_paths)
        elif isinstance(max_samples_per_dataset, int):
            # Single int -> apply to all datasets
            max_samples_list = [max_samples_per_dataset] * len(dataset_paths)
        else:
            max_samples_list = max_samples_per_dataset
            if len(max_samples_list) != len(dataset_paths):
                raise ValueError(
                    f"max_samples_per_dataset length ({len(max_samples_list)}) "
                    f"must match number of datasets ({len(dataset_paths)})"
                )
        
        # Load all datasets with proper caching
        datasets_list = []
        print("\n" + "="*80)
        print("Loading Datasets:")
        print("="*80)
        
        for idx, (path, max_samples) in enumerate(zip(dataset_paths, max_samples_list)):
            # Load dataset - use keep_in_memory=False to use disk cache
            dataset = load_dataset(
                'parquet',
                data_files=path,
                cache_dir=cache_dir,
                keep_in_memory=False,
            )
            # Extract the actual dataset from DatasetDict
            if 'train' in dataset:
                ds = dataset['train']
            else:
                ds = dataset[list(dataset.keys())[0]]
            
            original_len = len(ds)
            
            # Apply max_samples if specified
            if max_samples is not None and max_samples < original_len:
                # Use indices instead of shuffle+select to avoid copying data
                # This is much faster and uses cache better
                import random
                random.seed(42)
                indices = list(range(original_len))
                random.shuffle(indices)
                selected_indices = indices[:max_samples]
                ds = ds.select(selected_indices)
                selected_len = len(ds)
                print(f"  [{idx+1}] {path}")
                print(f"      Original: {original_len:,} samples")
                print(f"      Selected: {selected_len:,} samples (max_samples={max_samples:,})")
            else:
                selected_len = original_len
                print(f"  [{idx+1}] {path}")
                print(f"      Samples: {selected_len:,}")
            
            datasets_list.append(ds)
        
        # Concatenate if multiple datasets
        if len(datasets_list) == 1:
            self.dataset = datasets_list[0]
            print("="*80 + "\n")
        else:
            # Use concatenate_datasets with proper caching
            # Setting axis=0 ensures row-wise concatenation (default)
            print("-" * 80)
            print(f"Concatenating {len(datasets_list)} datasets...")
            self.dataset = concatenate_datasets(datasets_list)
            print(f"Total: {len(self.dataset):,} samples")
            print("="*80 + "\n")
        
        # Log filtering thresholds if enabled
        if self.max_atoms is not None or self.max_edges is not None:
            print("Runtime Filtering Thresholds:")
            if self.max_atoms is not None:
                print(f"  Max atoms: {self.max_atoms:,}")
            if self.max_edges is not None:
                print(f"  Max edges: {self.max_edges:,}")
            print(f"  Skip on error: {self.skip_on_error}")
            print("="*80 + "\n")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def get_filter_stats(self) -> Dict[str, int]:
        """Get statistics about filtered samples."""
        return {
            'filtered_count': self._filtered_count,
            'error_count': self._error_count,
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single graph sample with optional runtime filtering.
        
        Returns:
            Dictionary with 'modality' and 'value' keys matching AAEmbedder input format
        
        Raises:
            RuntimeError: If sample exceeds thresholds and skip_on_error=False
            ValueError: If sample data is invalid and skip_on_error=False
        
        Note:
            When skip_on_error=True and a sample is filtered, this will recursively
            try the next sample. This may cause issues with distributed training
            or if many consecutive samples are filtered.
        """
        try:
            item = self.dataset[idx]
            modality = item['modality']
            
            # Early check: count atoms and edges before tensor conversion
            if self.max_atoms is not None or self.max_edges is not None:
                # Count atoms
                num_atoms = len(item['node_feat']) if 'node_feat' in item else 0
                
                # Count edges - need to check multiple possible edge sources
                num_edges = 0
                if 'edge_index' in item and item['edge_index'] is not None:
                    edge_index_data = item['edge_index']
                    # edge_index is [2, E], so length gives us E
                    if isinstance(edge_index_data, list) and len(edge_index_data) > 0:
                        num_edges = len(edge_index_data[0]) if len(edge_index_data) > 0 else 0
                
                # For molecules, also count chemical edges
                if modality == 'molecule' and 'chem_edge_index' in item and item['chem_edge_index'] is not None:
                    chem_edge_data = item['chem_edge_index']
                    if isinstance(chem_edge_data, list) and len(chem_edge_data) > 0:
                        num_edges += len(chem_edge_data[0]) if len(chem_edge_data) > 0 else 0
                
                # Check thresholds
                if self.max_atoms is not None and num_atoms > self.max_atoms:
                    self._filtered_count += 1
                    if self.skip_on_error:
                        # Try next sample (with wraparound)
                        next_idx = (idx + 1) % len(self.dataset)
                        if next_idx == idx:
                            raise RuntimeError(f"All samples filtered - atom threshold too strict")
                        return self.__getitem__(next_idx)
                    else:
                        raise RuntimeError(
                            f"Sample {idx} exceeds max_atoms threshold: "
                            f"{num_atoms} atoms > {self.max_atoms} (modality: {modality})"
                        )
                
                if self.max_edges is not None and num_edges > self.max_edges:
                    self._filtered_count += 1
                    if self.skip_on_error:
                        # Try next sample (with wraparound)
                        next_idx = (idx + 1) % len(self.dataset)
                        if next_idx == idx:
                            raise RuntimeError(f"All samples filtered - edge threshold too strict")
                        return self.__getitem__(next_idx)
                    else:
                        raise RuntimeError(
                            f"Sample {idx} exceeds max_edges threshold: "
                            f"{num_edges} edges > {self.max_edges} (modality: {modality})"
                        )
            
            # Convert lists to tensors
            value = {}
            
            # Common fields
            value['node_feat'] = torch.tensor(item['node_feat'], dtype=torch.float32)
            
            # Handle coordinate field - support both 'pos' and 'coordinates' for compatibility
            # Molecules use 'pos', while protein/DNA/RNA datasets use 'coordinates'
            if 'pos' in item and item['pos'] is not None:
                value['pos'] = torch.tensor(item['pos'], dtype=torch.float32)
            elif 'coordinates' in item and item['coordinates'] is not None:
                value['pos'] = torch.tensor(item['coordinates'], dtype=torch.float32)
            else:
                raise KeyError(f"Missing coordinate field ('pos' or 'coordinates') for modality '{modality}' at index {idx}")
            
            value['edge_index'] = torch.tensor(item['edge_index'], dtype=torch.long)
            
            # Modality-specific fields
            if modality in ['protein', 'dna', 'rna']:
                # Protein and nucleic acids use distance-based edges
                # Support both unified format (edge_feat_dist) and legacy format (edge_attr)
                if 'edge_feat_dist' in item and item['edge_feat_dist'] is not None:
                    value['edge_feat_dist'] = torch.tensor(item['edge_feat_dist'], dtype=torch.float32)
                    if value['edge_feat_dist'].dim() == 1:
                        value['edge_feat_dist'] = value['edge_feat_dist'].unsqueeze(-1)
                elif 'edge_attr' in item and item['edge_attr'] is not None:
                    # Legacy format: also store as edge_attr for backward compatibility
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
        
        except Exception as e:
            self._error_count += 1
            if self.skip_on_error:
                # Try next sample (with wraparound)
                next_idx = (idx + 1) % len(self.dataset)
                if next_idx == idx:
                    raise RuntimeError(f"All samples failed to load") from e
                # Only log first few errors to avoid spam
                if self._error_count <= 5:
                    print(f"Warning: Error loading sample {idx}: {e}. Skipping to next sample.")
                return self.__getitem__(next_idx)
            else:
                raise


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
            max_patch_frac=0.10,
            accept_p=0.85,
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
        # (This is typical for training - separate dataloaders for protein/molecule/nucleic acid)
        if len(set(modalities)) > 1:
            raise ValueError("All graphs in batch must have the same modality")
        
        modality = modalities[0]
        
        # Extract value dicts
        graphs = [item['value'] for item in batch_list]
        
        # Merge graphs based on modality
        if modality == 'protein':
            merged = merge_protein_graphs(graphs)
        elif modality in ['dna', 'rna']:
            merged = merge_nucleic_acid_graphs(graphs)
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


class ModalityAwareBatchSampler(Sampler):
    """
    Batch sampler that ensures all samples in a batch have the same modality.
    
    This is required because the GraphBatchCollator enforces that all graphs in a batch
    must have the same modality. This sampler pre-groups samples by modality and creates
    batches within each modality group.
    
    Args:
        dataset: GraphDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle samples within each modality group
        seed: Random seed for shuffling
        drop_last: Whether to drop the last incomplete batch
    """
    
    def __init__(
        self,
        dataset: GraphDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        
        # Group indices by modality
        self.modality_indices = {}
        print("\nGrouping dataset by modality...")
        
        # Optimized: Access raw HuggingFace dataset to avoid tensor conversion
        try:
            # Handle torch.utils.data.Subset wrapping GraphDataset
            if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
                # This is a Subset object
                base_dataset = dataset.dataset  # GraphDataset
                subset_indices = dataset.indices
                
                if hasattr(base_dataset, 'dataset'):
                    # Access the HF dataset inside GraphDataset
                    raw_dataset = base_dataset.dataset
                    
                    # OPTIMIZED: Bulk read all modalities at once, then index
                    all_modalities = raw_dataset['modality']
                    
                    # Fast grouping: only access modality for subset indices
                    for local_idx, global_idx in enumerate(subset_indices):
                        modality = all_modalities[global_idx]  # Direct list access, very fast
                        if modality not in self.modality_indices:
                            self.modality_indices[modality] = []
                        self.modality_indices[modality].append(local_idx)
                else:
                    raise AttributeError("Base dataset doesn't have 'dataset' attribute")
                    
            # Handle GraphDataset directly (not wrapped in Subset)
            elif hasattr(dataset, 'dataset'):
                raw_dataset = dataset.dataset
                modalities = raw_dataset['modality']
                
                # Fast grouping using list comprehension
                for idx, modality in enumerate(modalities):
                    if modality not in self.modality_indices:
                        self.modality_indices[modality] = []
                    self.modality_indices[modality].append(idx)
            else:
                raise AttributeError("Unknown dataset type")
                
        except Exception as e:
            # Fallback: This should rarely be needed now
            print(f"Warning: Fast grouping failed ({e}), using fallback method...")
            dataset_len = len(dataset)
            
            # Track which samples have errors
            error_count = 0
            for idx in range(dataset_len):
                try:
                    # Try to get just the raw data without tensor conversion
                    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
                        # Subset: access base dataset's raw data
                        base_idx = dataset.indices[idx]
                        if hasattr(dataset.dataset, 'dataset'):
                            # GraphDataset wrapping HF dataset
                            raw_item = dataset.dataset.dataset[base_idx]
                        else:
                            raw_item = dataset.dataset[base_idx]
                        modality = raw_item['modality']
                    elif hasattr(dataset, 'dataset'):
                        # Direct GraphDataset access
                        modality = dataset.dataset[idx]['modality']
                    else:
                        # Last resort: use __getitem__ (slow and may fail for molecules)
                        sample = dataset[idx]
                        modality = sample['modality']
                    
                    if modality not in self.modality_indices:
                        self.modality_indices[modality] = []
                    self.modality_indices[modality].append(idx)
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Only print first 5 errors
                        print(f"Warning: Could not access sample {idx}: {e}")
                    continue
            
            if error_count > 5:
                print(f"Warning: {error_count} total samples could not be accessed")
        
        # Log modality distribution
        print("-" * 80)
        for modality, indices in self.modality_indices.items():
            print(f"  {modality}: {len(indices):,} samples")
        print("-" * 80 + "\n")
        
        self.epoch = 0
    
    def __iter__(self):
        # Set random seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Shuffle indices within each modality if needed
        all_batches = []
        for modality, indices in self.modality_indices.items():
            indices_copy = indices.copy()
            
            if self.shuffle:
                # Shuffle within modality
                random.Random(self.seed + self.epoch).shuffle(indices_copy)
            
            # Create batches for this modality
            for i in range(0, len(indices_copy), self.batch_size):
                batch = indices_copy[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle the order of batches (so different modalities are interleaved)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        total_batches = 0
        for indices in self.modality_indices.values():
            num_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                num_batches += 1
            total_batches += num_batches
        return total_batches
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch

