"""
Adaptive EGNN with Automatic Coarse-Graining for Large Graphs

This module provides an adaptive wrapper around EGNN that automatically applies
coarse-graining when graphs exceed memory-safe thresholds, while processing
smaller graphs at full atomic resolution.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Literal
from .egnn import EGNNBackbone


class AdaptiveEGNN(nn.Module):
    """
    Adaptive EGNN that automatically coarse-grains large graphs.
    
    Strategy:
    - Small graphs (< threshold): Process at full atomic resolution
    - Large graphs (>= threshold): Apply coarse-graining before processing
    
    Coarse-graining options:
    - 'backbone': Keep only backbone atoms (10-40x reduction for nucleic acids/proteins)
    - 'per_residue': Pool atoms within each residue to single node (40-60x reduction)
    - 'downsample': Randomly downsample to target size (configurable reduction)
    - 'adaptive': Choose strategy based on modality and size
    
    Args:
        egnn_backbone: EGNN backbone module to wrap
        atom_threshold: Max atoms before coarse-graining (default: 100,000)
        edge_threshold: Max edges before coarse-graining (default: 2,000,000)
        coarse_strategy: Coarse-graining strategy ('backbone', 'per_residue', 'downsample', 'adaptive')
        restore_full_resolution: Whether to project back to full resolution (default: False)
        downsample_target: Target number of atoms for downsampling (default: 5000)
    """
    
    def __init__(
        self,
        egnn_backbone: EGNNBackbone,
        atom_threshold: int = 100_000,
        edge_threshold: int = 2_000_000,
        coarse_strategy: Literal['backbone', 'per_residue', 'downsample', 'adaptive'] = 'backbone',
        restore_full_resolution: bool = False,
        downsample_target: int = 5000,
    ):
        super().__init__()
        self.egnn = egnn_backbone
        self.atom_threshold = atom_threshold
        self.edge_threshold = edge_threshold
        self.coarse_strategy = coarse_strategy
        self.restore_full_resolution = restore_full_resolution
        self.downsample_target = downsample_target
        
        # Statistics tracking
        self.register_buffer('num_coarsened', torch.tensor(0, dtype=torch.long))
        self.register_buffer('num_full_resolution', torch.tensor(0, dtype=torch.long))
        
        # Upsampling network (if restore_full_resolution=True)
        if restore_full_resolution:
            dim = egnn_backbone.dim
            self.upsample_net = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim)
            )
    
    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
        node_feat: Optional[torch.Tensor] = None,
        modality: Optional[str] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive forward pass with automatic coarse-graining.
        
        Args:
            h: Node embeddings [N, dim]
            pos: Node coordinates [N, 3]
            edge_index: Edge indices [2, E]
            e: Edge embeddings [E, dim]
            node_feat: Optional raw node features [N, num_feat] for coarse-graining
            modality: Optional modality hint ('protein', 'dna', 'rna', 'molecule')
            batch: Optional batch assignment [N]
        
        Returns:
            h_out: Output node embeddings [N, dim] (or coarsened size)
            pos_out: Output coordinates [N, 3] (or coarsened size)
        """
        num_atoms = h.size(0)
        num_edges = edge_index.size(1)
        
        # Check if coarse-graining is needed
        needs_coarsening = (
            num_atoms > self.atom_threshold or 
            num_edges > self.edge_threshold
        )
        
        if not needs_coarsening:
            # Process at full resolution
            self.num_full_resolution += 1
            return self.egnn(h, pos, edge_index, e, batch)
        
        # Apply coarse-graining
        self.num_coarsened += 1
        
        # Log coarse-graining info (only occasionally to avoid spam)
        if self.num_coarsened % 10 == 1:  # Log every 10th coarsening
            print(f"[AdaptiveEGNN] Coarse-graining: atoms={num_atoms:,} (threshold={self.atom_threshold:,}), "
                  f"edges={num_edges:,} (threshold={self.edge_threshold:,}), "
                  f"modality={modality}, strategy={self.coarse_strategy}")
        
        # Determine strategy
        strategy = self.coarse_strategy
        if strategy == 'adaptive':
            strategy = self._choose_strategy(modality, num_atoms)
        
        # Coarse-grain the graph
        coarse_data = self._coarse_grain(
            h=h,
            pos=pos,
            edge_index=edge_index,
            e=e,
            node_feat=node_feat,
            modality=modality,
            batch=batch,
            strategy=strategy
        )
        
        if coarse_data is None:
            # Fallback to full resolution if coarse-graining fails
            if self.num_coarsened % 10 == 1:
                print(f"[AdaptiveEGNN] WARNING: Coarse-graining failed (node_feat available: {node_feat is not None}), "
                      f"falling back to full resolution. This may cause OOM!")
            return self.egnn(h, pos, edge_index, e, batch)
        
        # Log reduction factor
        if self.num_coarsened % 10 == 1:
            reduction = coarse_data.get('reduction_factor', 1.0)
            print(f"[AdaptiveEGNN] Reduced: {num_atoms:,} → {coarse_data['h'].size(0):,} atoms "
                  f"({reduction:.1f}x reduction), edges: {num_edges:,} → {coarse_data['edge_index'].size(1):,}")
        
        # Process coarsened graph
        h_coarse, pos_coarse = self.egnn(
            coarse_data['h'],
            coarse_data['pos'],
            coarse_data['edge_index'],
            coarse_data['e'],
            coarse_data.get('batch')
        )
        
        # Optionally restore to full resolution
        if self.restore_full_resolution and coarse_data.get('mapping') is not None:
            h_full = self._upsample(h_coarse, coarse_data['mapping'], num_atoms)
            pos_full = self._upsample_coords(pos_coarse, coarse_data['mapping'], num_atoms)
            return h_full, pos_full
        
        return h_coarse, pos_coarse
    
    def _choose_strategy(self, modality: Optional[str], num_atoms: int) -> str:
        """Choose coarse-graining strategy based on modality and size."""
        if num_atoms > 50_000:
            # For very large molecules, use downsampling
            return 'downsample'
        elif modality in ['protein', 'dna', 'rna']:
            # Backbone strategy for medium-sized biomolecules
            return 'backbone'
        else:
            # Per-residue for molecules
            return 'per_residue' if num_atoms > 20_000 else 'backbone'
    
    def _coarse_grain(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
        node_feat: Optional[torch.Tensor],
        modality: Optional[str],
        batch: Optional[torch.Tensor],
        strategy: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Apply coarse-graining to the graph."""
        
        if strategy == 'backbone':
            return self._coarse_grain_backbone(
                h, pos, edge_index, e, node_feat, batch
            )
        elif strategy == 'per_residue':
            return self._coarse_grain_per_residue(
                h, pos, edge_index, e, node_feat, batch
            )
        elif strategy == 'downsample':
            return self._coarse_grain_downsample(
                h, pos, edge_index, e, node_feat, batch
            )
        else:
            return None
    
    def _coarse_grain_backbone(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
        node_feat: Optional[torch.Tensor],
        batch: Optional[torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Coarse-grain by keeping only backbone atoms.
        
        For proteins: is_backbone flag at index 5
        For nucleic acids: is_backbone flag at index 5
        For molecules: no backbone concept, skip coarse-graining
        """
        if node_feat is None:
            # Can't apply backbone filtering without node features
            return None
        
        # Detect backbone atoms (feature index 5 should be is_backbone)
        if node_feat.size(1) < 6:
            # Not enough features for backbone flag
            return None
        
        backbone_mask = node_feat[:, 5] == 1
        
        if backbone_mask.sum() == 0:
            # No backbone atoms found (e.g., small molecule)
            return None
        
        # Filter nodes
        h_coarse = h[backbone_mask]
        pos_coarse = pos[backbone_mask]
        batch_coarse = batch[backbone_mask] if batch is not None else None
        
        # Create mapping from original to coarse indices
        old_to_new = torch.full((h.size(0),), -1, dtype=torch.long, device=h.device)
        old_to_new[backbone_mask] = torch.arange(
            backbone_mask.sum(), dtype=torch.long, device=h.device
        )
        
        # Filter edges (keep only edges between backbone atoms)
        edge_mask = backbone_mask[edge_index[0]] & backbone_mask[edge_index[1]]
        edge_index_coarse = old_to_new[edge_index[:, edge_mask]]
        e_coarse = e[edge_mask]
        
        return {
            'h': h_coarse,
            'pos': pos_coarse,
            'edge_index': edge_index_coarse,
            'e': e_coarse,
            'batch': batch_coarse,
            'mapping': old_to_new,  # For upsampling if needed
            'reduction_factor': h.size(0) / h_coarse.size(0)
        }
    
    def _coarse_grain_per_residue(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
        node_feat: Optional[torch.Tensor],
        batch: Optional[torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Coarse-grain by pooling atoms within each residue.
        
        Uses residue_id at feature index 4 to group atoms.
        """
        if node_feat is None or node_feat.size(1) < 5:
            # Can't apply per-residue pooling without residue_id
            return None
        
        # Get residue IDs (feature index 4)
        residue_ids = node_feat[:, 4].long()
        
        # If batch is provided, make residue IDs unique per batch
        if batch is not None:
            max_residue = residue_ids.max() + 1
            residue_ids = residue_ids + batch * max_residue
        
        # Get unique residues and create mapping
        unique_residues, inverse_indices = torch.unique(residue_ids, return_inverse=True)
        num_coarse = unique_residues.size(0)
        
        # Pool node features and coordinates by residue (mean pooling)
        h_coarse = torch.zeros(num_coarse, h.size(1), dtype=h.dtype, device=h.device)
        pos_coarse = torch.zeros(num_coarse, 3, dtype=pos.dtype, device=pos.device)
        
        h_coarse.index_add_(0, inverse_indices, h)
        pos_coarse.index_add_(0, inverse_indices, pos)
        
        # Normalize by counts
        counts = torch.bincount(inverse_indices, minlength=num_coarse).float()
        h_coarse = h_coarse / counts.unsqueeze(1).clamp(min=1)
        pos_coarse = pos_coarse / counts.unsqueeze(1).clamp(min=1)
        
        # Build coarse edges (connect residues that had inter-atom edges)
        src_residues = inverse_indices[edge_index[0]]
        dst_residues = inverse_indices[edge_index[1]]
        
        # Remove self-loops and get unique edges
        edge_pairs = torch.stack([src_residues, dst_residues], dim=0)
        edge_pairs_unique = torch.unique(edge_pairs, dim=1)
        
        # Remove self-loops
        mask = edge_pairs_unique[0] != edge_pairs_unique[1]
        edge_index_coarse = edge_pairs_unique[:, mask]
        
        # Edge features: use mean of edges between the same residue pairs
        # For simplicity, create uniform edge features (could be improved)
        e_coarse = torch.zeros(
            edge_index_coarse.size(1), e.size(1), 
            dtype=e.dtype, device=e.device
        )
        
        # Pool batch assignment
        if batch is not None:
            batch_coarse = torch.zeros(num_coarse, dtype=batch.dtype, device=batch.device)
            batch_coarse.index_copy_(0, inverse_indices, batch)
        else:
            batch_coarse = None
        
        return {
            'h': h_coarse,
            'pos': pos_coarse,
            'edge_index': edge_index_coarse,
            'e': e_coarse,
            'batch': batch_coarse,
            'mapping': inverse_indices,  # Maps original nodes to coarse nodes
            'reduction_factor': h.size(0) / h_coarse.size(0)
        }
    
    def _coarse_grain_downsample(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
        node_feat: Optional[torch.Tensor],
        batch: Optional[torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Coarse-grain by randomly downsampling nodes to target size.
        
        This is the most aggressive strategy - randomly keeps a subset of atoms
        to fit within memory constraints. Good for very large molecules.
        
        Strategy:
        - Randomly select `downsample_target` nodes (or fewer if graph is smaller)
        - Keep edges between selected nodes
        - Very aggressive: can achieve 10x-100x reduction
        """
        num_atoms = h.size(0)
        
        # Determine target size (don't upsample if already small)
        target_size = min(self.downsample_target, num_atoms)
        
        if target_size >= num_atoms:
            # Graph is already small enough
            return None
        
        # Randomly sample nodes (uniform sampling)
        # Use deterministic sampling based on batch to ensure consistency
        if batch is not None:
            # Sample proportionally from each graph in batch
            unique_batches = torch.unique(batch)
            sampled_indices = []
            
            for b in unique_batches:
                batch_mask = batch == b
                batch_size = batch_mask.sum().item()
                batch_target = max(1, int(target_size * batch_size / num_atoms))
                
                batch_indices = torch.where(batch_mask)[0]
                perm = torch.randperm(batch_indices.size(0), device=h.device)[:batch_target]
                sampled_indices.append(batch_indices[perm])
            
            keep_mask = torch.zeros(num_atoms, dtype=torch.bool, device=h.device)
            for indices in sampled_indices:
                keep_mask[indices] = True
        else:
            # Simple random sampling
            perm = torch.randperm(num_atoms, device=h.device)[:target_size]
            keep_mask = torch.zeros(num_atoms, dtype=torch.bool, device=h.device)
            keep_mask[perm] = True
        
        # Filter nodes
        h_coarse = h[keep_mask]
        pos_coarse = pos[keep_mask]
        batch_coarse = batch[keep_mask] if batch is not None else None
        
        # Create mapping from original to coarse indices
        old_to_new = torch.full((num_atoms,), -1, dtype=torch.long, device=h.device)
        old_to_new[keep_mask] = torch.arange(
            keep_mask.sum(), dtype=torch.long, device=h.device
        )
        
        # Filter edges (keep only edges between sampled nodes)
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        edge_index_coarse = old_to_new[edge_index[:, edge_mask]]
        e_coarse = e[edge_mask]
        
        return {
            'h': h_coarse,
            'pos': pos_coarse,
            'edge_index': edge_index_coarse,
            'e': e_coarse,
            'batch': batch_coarse,
            'mapping': old_to_new,
            'reduction_factor': num_atoms / h_coarse.size(0)
        }
    
    def _upsample(
        self,
        h_coarse: torch.Tensor,
        mapping: torch.Tensor,
        target_size: int
    ) -> torch.Tensor:
        """Upsample coarse features back to full resolution."""
        h_full = torch.zeros(target_size, h_coarse.size(1), 
                            dtype=h_coarse.dtype, device=h_coarse.device)
        
        # Map coarse features back to original nodes
        valid_mask = mapping >= 0
        h_full[valid_mask] = h_coarse[mapping[valid_mask]]
        
        # Apply upsampling network
        h_full = self.upsample_net(h_full)
        
        return h_full
    
    def _upsample_coords(
        self,
        pos_coarse: torch.Tensor,
        mapping: torch.Tensor,
        target_size: int
    ) -> torch.Tensor:
        """Upsample coarse coordinates back to full resolution."""
        pos_full = torch.zeros(target_size, 3, 
                              dtype=pos_coarse.dtype, device=pos_coarse.device)
        
        # Map coarse coords back to original nodes
        valid_mask = mapping >= 0
        pos_full[valid_mask] = pos_coarse[mapping[valid_mask]]
        
        return pos_full
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics on coarse-graining usage."""
        return {
            'num_coarsened': self.num_coarsened.item(),
            'num_full_resolution': self.num_full_resolution.item(),
            'coarsening_rate': (
                self.num_coarsened.item() / 
                max(1, self.num_coarsened.item() + self.num_full_resolution.item())
            )
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.num_coarsened.zero_()
        self.num_full_resolution.zero_()


def create_adaptive_egnn(
    dim: int = 256,
    num_layers: int = 6,
    dropout: float = 0.1,
    atom_threshold: int = 100_000,
    edge_threshold: int = 2_000_000,
    coarse_strategy: Literal['backbone', 'per_residue', 'downsample', 'adaptive'] = 'backbone',
    downsample_target: int = 5000,
    **egnn_kwargs
) -> AdaptiveEGNN:
    """
    Factory function to create an AdaptiveEGNN.
    
    Args:
        dim: Hidden dimension
        num_layers: Number of EGNN layers
        dropout: Dropout rate
        atom_threshold: Max atoms before coarse-graining
        edge_threshold: Max edges before coarse-graining
        coarse_strategy: Coarse-graining strategy
        downsample_target: Target size for downsampling strategy
        **egnn_kwargs: Additional arguments for EGNNBackbone
    
    Returns:
        AdaptiveEGNN module
    
    Example:
        >>> model = create_adaptive_egnn(
        ...     dim=256,
        ...     num_layers=6,
        ...     atom_threshold=10_000,
        ...     coarse_strategy='downsample',
        ...     downsample_target=5000
        ... )
    """
    egnn = EGNNBackbone(
        dim=dim,
        num_layers=num_layers,
        dropout=dropout,
        **egnn_kwargs
    )
    
    return AdaptiveEGNN(
        egnn_backbone=egnn,
        atom_threshold=atom_threshold,
        edge_threshold=edge_threshold,
        coarse_strategy=coarse_strategy,
        downsample_target=downsample_target,
    )

