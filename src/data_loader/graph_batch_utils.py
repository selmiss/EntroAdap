"""
Graph Batching Utilities

Helper functions for merging and batching graph structures.
"""

import torch
import math
from typing import Dict, List, Set, Optional
from collections import deque

def merge_protein_graphs(graphs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Merge protein graphs into a batch.
    
    Args:
        graphs: List of protein graph dictionaries
    
    Returns:
        Merged batch dictionary with node offset applied
    """
    batch_node_feat = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_pos = []
    batch_indices = []
    
    node_offset = 0
    for graph_idx, g in enumerate(graphs):
        num_nodes = g['node_feat'].size(0)
        
        batch_node_feat.append(g['node_feat'])
        batch_pos.append(g['pos'])
        batch_indices.extend([graph_idx] * num_nodes)
        
        # Offset edge indices
        edge_index = g['edge_index'] + node_offset
        batch_edge_index.append(edge_index)
        
        if 'edge_attr' in g:
            batch_edge_attr.append(g['edge_attr'])
        
        node_offset += num_nodes
    
    merged = {
        'node_feat': torch.cat(batch_node_feat, dim=0),
        'pos': torch.cat(batch_pos, dim=0),
        'edge_index': torch.cat(batch_edge_index, dim=1),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
    }
    
    if batch_edge_attr:
        merged['edge_attr'] = torch.cat(batch_edge_attr, dim=0)
    
    return merged


def merge_nucleic_acid_graphs(graphs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Merge nucleic acid (DNA/RNA) graphs into a batch.
    
    Similar to protein graph merging (both use distance-based edges only).
    
    Args:
        graphs: List of nucleic acid graph dictionaries
    
    Returns:
        Merged batch dictionary with node offset applied
    """
    batch_node_feat = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_pos = []
    batch_indices = []
    
    node_offset = 0
    for graph_idx, g in enumerate(graphs):
        num_nodes = g['node_feat'].size(0)
        
        batch_node_feat.append(g['node_feat'])
        batch_pos.append(g['pos'])
        batch_indices.extend([graph_idx] * num_nodes)
        
        # Offset edge indices
        edge_index = g['edge_index'] + node_offset
        batch_edge_index.append(edge_index)
        
        if 'edge_attr' in g:
            batch_edge_attr.append(g['edge_attr'])
        
        node_offset += num_nodes
    
    merged = {
        'node_feat': torch.cat(batch_node_feat, dim=0),
        'pos': torch.cat(batch_pos, dim=0),
        'edge_index': torch.cat(batch_edge_index, dim=1),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
    }
    
    if batch_edge_attr:
        merged['edge_attr'] = torch.cat(batch_edge_attr, dim=0)
    
    return merged


def merge_molecule_graphs(graphs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Merge molecule graphs into a batch.
    
    Args:
        graphs: List of molecule graph dictionaries
    
    Returns:
        Merged batch dictionary with node offset applied
    """
    batch_node_feat = []
    batch_edge_index = []
    batch_edge_feat_dist = []
    batch_chem_edge_index = []
    batch_chem_edge_feat = []
    batch_pos = []
    batch_indices = []
    
    node_offset = 0
    for graph_idx, g in enumerate(graphs):
        num_nodes = g['node_feat'].size(0)
        
        batch_node_feat.append(g['node_feat'])
        batch_pos.append(g['pos'])
        batch_indices.extend([graph_idx] * num_nodes)
        
        # Spatial edges
        if 'edge_index' in g:
            edge_index = g['edge_index'] + node_offset
            batch_edge_index.append(edge_index)
            if 'edge_feat_dist' in g:
                batch_edge_feat_dist.append(g['edge_feat_dist'])
        
        # Chemical edges
        if 'chem_edge_index' in g:
            chem_edge_index = g['chem_edge_index'] + node_offset
            batch_chem_edge_index.append(chem_edge_index)
            if 'chem_edge_feat_cat' in g:
                batch_chem_edge_feat.append(g['chem_edge_feat_cat'])
        
        node_offset += num_nodes
    
    merged = {
        'node_feat': torch.cat(batch_node_feat, dim=0),
        'pos': torch.cat(batch_pos, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
    }
    
    if batch_edge_index:
        merged['edge_index'] = torch.cat(batch_edge_index, dim=1)
    if batch_edge_feat_dist:
        merged['edge_feat_dist'] = torch.cat(batch_edge_feat_dist, dim=0)
    
    if batch_chem_edge_index:
        merged['chem_edge_index'] = torch.cat(batch_chem_edge_index, dim=1)
    if batch_chem_edge_feat:
        merged['chem_edge_feat_cat'] = torch.cat(batch_chem_edge_feat, dim=0)
    
    return merged


def bfs_patch_masking(
    edge_index: torch.Tensor,
    num_nodes: int,
    target_mask_ratio: float,
    *,
    min_patch_size: int = 1,
    max_patch_frac: float = 0.02,
    accept_p: float = 0.7,
    force_fill_to_target: bool = True,
    make_undirected: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Set[int]:
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    assert 0.0 < target_mask_ratio < 1.0
    assert 0.0 <= accept_p <= 1.0
    assert min_patch_size >= 1
    assert max_patch_frac > 0.0

    if generator is None:
        generator = torch.Generator(device="cpu")

    target_num_masked = int(round(num_nodes * target_mask_ratio))
    target_num_masked = max(0, min(target_num_masked, num_nodes))
    # Ensure at least one node is masked if nodes are available
    if num_nodes > 0:
        target_num_masked = max(1, target_num_masked)
    if target_num_masked == 0:
        return set()

    ei = edge_index.detach()
    if ei.is_cuda:
        ei = ei.cpu()

    if make_undirected:
        ei = torch.cat([ei, ei.flip(0)], dim=1)

    src = ei[0].long()
    dst = ei[1].long()

    valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
    src, dst = src[valid], dst[valid]

    perm = torch.argsort(src)
    src, dst = src[perm], dst[perm]

    deg = torch.bincount(src, minlength=num_nodes)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    indptr[1:] = torch.cumsum(deg, dim=0)
    neighbors = dst

    masked = torch.zeros(num_nodes, dtype=torch.bool)
    available = torch.ones(num_nodes, dtype=torch.bool)

    patch_cap = int(math.ceil(num_nodes * max_patch_frac))
    patch_cap = max(1, patch_cap)

    while int(masked.sum().item()) < target_num_masked and bool(available.any().item()):
        remaining = target_num_masked - int(masked.sum().item())

        avail_idx = torch.nonzero(available, as_tuple=False).view(-1)
        seed = int(avail_idx[torch.randint(len(avail_idx), (1,), generator=generator)].item())
        available[seed] = False

        # 每个 patch 的预算严格不超过剩余预算
        hi = min(patch_cap, remaining)
        lo = min(min_patch_size, hi)
        patch_budget = int(torch.randint(lo, hi + 1, (1,), generator=generator).item())

        queue = deque([seed])
        patch = {seed}

        # BFS 扩展时也要随时检查全局预算，禁止超标
        while queue and len(patch) < patch_budget and (len(patch) + int(masked.sum().item()) < target_num_masked):
            u = queue.popleft()
            start = int(indptr[u].item())
            end = int(indptr[u + 1].item())
            if start == end:
                continue

            # 遍历邻居
            for v in neighbors[start:end].tolist():
                if masked[v]:
                    continue
                if v in patch:
                    continue

                if len(patch) + int(masked.sum().item()) >= target_num_masked:
                    break
                if len(patch) >= patch_budget:
                    break

                take = bool(torch.rand((), generator=generator).item() < accept_p)
                if take:
                    patch.add(v)
                    queue.append(v)
                    available[v] = False

        # commit，保证不会超过 target
        for v in patch:
            if int(masked.sum().item()) >= target_num_masked:
                break
            masked[v] = True
            available[v] = False

    # 可选补齐，补齐也严格不超过 target
    if force_fill_to_target and int(masked.sum().item()) < target_num_masked:
        remaining = target_num_masked - int(masked.sum().item())
        cand = torch.nonzero(~masked, as_tuple=False).view(-1)
        if len(cand) > 0:
            k = min(remaining, len(cand))
            pick = cand[torch.randperm(len(cand), generator=generator)[:k]]
            masked[pick] = True

    return set(torch.nonzero(masked, as_tuple=False).view(-1).tolist())
