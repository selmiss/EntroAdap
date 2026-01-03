"""
All-Atom Embedder for Molecular, Protein, and Nucleic Acid Graphs

Compact embedder that handles protein, molecule, and nucleic acid (DNA/RNA) features with automatic detection.
Uses offset-based embedding tables for categorical features and RBF for continuous features.

All embeddings are projected to a consistent hidden_dim through linear layers, ensuring
uniform dimensionality for downstream processing (e.g., EGNN layers).
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class RBFExpansion(nn.Module):
    """Radial Basis Function expansion for distance features."""
    
    def __init__(self, num_rbf: int = 32, rbf_min: float = 0.0, rbf_max: float = 10.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(rbf_min, rbf_max, num_rbf))
        self.gamma = 1.0 / ((rbf_max - rbf_min) / num_rbf) ** 2
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """distances: [E] or [E, 1] -> [E, num_rbf]"""
        if distances.dim() == 2:
            distances = distances.squeeze(-1)
        diff = distances.unsqueeze(-1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * diff ** 2)


class AAEmbedder(nn.Module):
    """
    Unified all-atom feature embedder using offset-based embedding for categorical features.
    
    Protein node features (7): [atomic_number(119), atom_name(46), residue(24), chain(27), residue_id(cont), is_backbone(2), is_ca(2)]
    Nucleic acid node features (7): [atomic_number(119), atom_name(30), nucleotide(11), chain(27), residue_id(cont), is_backbone(2), is_phosphate(2)]
    Molecule node features (9): [atomic_num(119), chirality(4), degree(12), charge(12), numH(10), radical(6), hybrid(6), aromatic(2), ring(2)]
    Protein edges: distance (float)
    Nucleic acid edges: distance (float)
    Molecule chem edges (3): [bond_type(5), bond_stereo(6), conjugated(2)]
    Molecule spatial edges: distance (float)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_rbf: int = 32,
        rbf_max: float = 10.0,
        protein_residue_id_scale: float = 1000.0,
        nucleic_acid_residue_id_scale: float = 500.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.protein_residue_id_scale = float(protein_residue_id_scale)
        self.nucleic_acid_residue_id_scale = float(nucleic_acid_residue_id_scale)
        
        # Protein node: 7 features with dims [119, 46, 24, 27, -1, 2, 2]
        # IMPORTANT: keep these in sync with src/data_factory/protein/cif_to_cooridinates.py (PROTEIN_ATOM_FEATURES).
        # Features: [atomic_number(119: 0-118), atom_name(46), residue(24), chain(27), residue_id(cont), is_backbone(2), is_ca(2)]
        # If the residue vocab is wrong, offset-based embedding will collide across fields.
        protein_dims = [119, 46, 24, 27, 2, 2]  # Skip continuous residue_id (index 4)
        protein_offset = torch.tensor([0] + protein_dims[:-1]).cumsum(0)
        self.register_buffer('protein_node_offset', protein_offset)
        self.protein_node_embed = nn.Embedding(sum(protein_dims), hidden_dim)
        self.protein_residue_proj = nn.Linear(1, hidden_dim)
        self.protein_node_combine = nn.Linear((len(protein_dims) + 1) * hidden_dim, hidden_dim)
        
        # Nucleic acid node: 7 features with dims [119, 30, 11, 27, -1, 2, 2]
        # IMPORTANT: keep these in sync with src/data_factory/nacid/seq_to_feature.py (ATOM_NAME_VOCAB, NUCLEOTIDE_VOCAB).
        # Features: [atomic_number(119: 0-118), atom_name(30), nucleotide(11), chain(27), residue_id(cont), is_backbone(2), is_phosphate(2)]
        nacid_dims = [119, 30, 11, 27, 2, 2]  # Skip continuous residue_id (index 4)
        nacid_offset = torch.tensor([0] + nacid_dims[:-1]).cumsum(0)
        self.register_buffer('nacid_node_offset', nacid_offset)
        self.nacid_node_embed = nn.Embedding(sum(nacid_dims), hidden_dim)
        self.nacid_residue_proj = nn.Linear(1, hidden_dim)
        self.nacid_node_combine = nn.Linear((len(nacid_dims) + 1) * hidden_dim, hidden_dim)
        
        # Molecule node: 9 features with dims [119, 4, 12, 12, 10, 6, 6, 2, 2]
        mol_dims = [119, 4, 12, 12, 10, 6, 6, 2, 2]
        mol_offset = torch.tensor([0] + mol_dims[:-1]).cumsum(0)
        self.register_buffer('mol_node_offset', mol_offset)
        self.mol_node_embed = nn.Embedding(sum(mol_dims), hidden_dim)
        self.mol_node_combine = nn.Linear(len(mol_dims) * hidden_dim, hidden_dim)
        
        # Molecule chem edge: 3 features with dims [5, 6, 2]
        chem_dims = [5, 6, 2]
        chem_offset = torch.tensor([0] + chem_dims[:-1]).cumsum(0)
        self.register_buffer('mol_chem_offset', chem_offset)
        self.mol_chem_embed = nn.Embedding(sum(chem_dims), hidden_dim)
        self.mol_chem_combine = nn.Linear(len(chem_dims) * hidden_dim, hidden_dim)
        
        # Distance embeddings (for both protein and molecule spatial edges)
        self.rbf = RBFExpansion(num_rbf, 0.0, rbf_max)
        self.dist_proj = nn.Linear(num_rbf, hidden_dim)
    
    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass that routes to the appropriate embedder based on modality.
        
        Args:
            data: Dictionary with keys:
                - 'modality': str, one of 'protein', 'molecule', 'dna', or 'rna'
                - 'value': Dict[str, torch.Tensor], the actual graph data
        
        Returns:
            Standardized graph dict with keys: node_emb, edge_emb, edge_index, pos
        """
        modality = data.get('modality', None)
        if modality is None:
            raise ValueError("Data must contain 'modality' key specifying 'protein' or 'molecule'")
        
        graph_data = data.get('value', None)
        if graph_data is None:
            raise ValueError("Data must contain 'value' key with the actual graph data")
        
        if modality == 'protein':
            return self.embed_protein_graph(graph_data)
        elif modality == 'molecule':
            return self.embed_molecule_graph(graph_data)
        elif modality in ['dna', 'rna']:
            return self.embed_nucleic_acid_graph(graph_data)
        else:
            raise ValueError(f"Unknown modality: {modality}. Expected 'protein', 'molecule', 'dna', or 'rna'")
    
    def embed_protein_graph(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Embed complete protein graph and return standardized format.
        
        Args:
            data: Dictionary with keys: node_feat, edge_attr, edge_index, pos
        
        Returns:
            Standardized graph dict with keys: node_emb, edge_emb, edge_index, pos
        """
        # Embed node features [N, 7]
        # Feature order: [atomic_number, atom_name, residue_name, chain, residue_id, is_backbone, is_ca]
        node_feat = data['node_feat']
        
        # Extract categorical features (indices 0, 1, 2, 3, 5, 6) - now including atomic_number
        cat_feats = node_feat[:, [0, 1, 2, 3, 5, 6]].long()
        
        # Extract continuous feature (residue_id at index 4)
        residue_id = node_feat[:, 4:5].float()
        if self.protein_residue_id_scale > 0:
            residue_id = residue_id / self.protein_residue_id_scale
        
        # Offset-based embedding for all categorical features including atomic_number
        offset_feats = cat_feats + self.protein_node_offset.unsqueeze(0)
        embeddings = self.protein_node_embed(offset_feats)  # [N, 6, hidden]
        
        # Project continuous feature
        residue_emb = self.protein_residue_proj(residue_id)  # [N, hidden]
        
        # Concatenate and combine
        h = torch.cat([embeddings.flatten(1), residue_emb], dim=-1)
        node_emb = self.protein_node_combine(h)
        
        # Embed edge features (distances)
        rbf_feats = self.rbf(data['edge_attr'])  # [E, num_rbf]
        edge_emb = self.dist_proj(rbf_feats)
        
        return {
            'node_emb': node_emb,
            'edge_emb': edge_emb,
            'edge_index': data['edge_index'],
            'pos': data['pos'],
        }
    
    def embed_nucleic_acid_graph(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Embed complete nucleic acid (DNA/RNA) graph and return standardized format.
        
        Similar to protein embedder but with different feature vocabularies.
        
        Args:
            data: Dictionary with keys: node_feat, edge_attr, edge_index, pos
        
        Returns:
            Standardized graph dict with keys: node_emb, edge_emb, edge_index, pos
        """
        # Embed node features [N, 7]
        # Feature order: [atomic_number, atom_name, nucleotide, chain, residue_id, is_backbone, is_phosphate]
        node_feat = data['node_feat']
        
        # Extract categorical features (indices 0, 1, 2, 3, 5, 6)
        cat_feats = node_feat[:, [0, 1, 2, 3, 5, 6]].long()
        
        # Extract continuous feature (residue_id at index 4)
        residue_id = node_feat[:, 4:5].float()
        if self.nucleic_acid_residue_id_scale > 0:
            residue_id = residue_id / self.nucleic_acid_residue_id_scale
        
        # Offset-based embedding for all categorical features
        offset_feats = cat_feats + self.nacid_node_offset.unsqueeze(0)
        embeddings = self.nacid_node_embed(offset_feats)  # [N, 6, hidden]
        
        # Project continuous feature
        residue_emb = self.nacid_residue_proj(residue_id)  # [N, hidden]
        
        # Concatenate and combine
        h = torch.cat([embeddings.flatten(1), residue_emb], dim=-1)
        node_emb = self.nacid_node_combine(h)
        
        # Embed edge features (distances)
        rbf_feats = self.rbf(data['edge_attr'])  # [E, num_rbf]
        edge_emb = self.dist_proj(rbf_feats)
        
        return {
            'node_emb': node_emb,
            'edge_emb': edge_emb,
            'edge_index': data['edge_index'],
            'pos': data['pos'],
        }
    
    def embed_molecule_graph(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Embed complete molecule graph and return standardized format.
        
        Handles concatenation of chemical and spatial edges internally.
        Both edge types share the same node set, so they can be concatenated directly.
        
        Args:
            data: Dictionary with keys:
                - node_feat: [N, 9] molecule features
                - pos: [N, 3] coordinates
                - edge_index: [2, E_spatial] spatial edges (optional)
                - edge_feat_dist: [E_spatial, 1] spatial distances (optional)
                - chem_edge_index: [2, E_chem] chemical edges (optional)
                - chem_edge_feat_cat: [E_chem, 3] chemical edge features (optional)
        
        Returns:
            Standardized graph dict with keys: node_emb, edge_emb, edge_index, pos
        """
        # Embed node features [N, 9]
        node_feat = data['node_feat']
        offset_feats = node_feat.long() + self.mol_node_offset.unsqueeze(0)
        embeddings = self.mol_node_embed(offset_feats)  # [N, 9, hidden]
        h = embeddings.flatten(1)
        node_emb = self.mol_node_combine(h)
        
        # Determine which edge types are available
        has_chem = 'chem_edge_feat_cat' in data and 'chem_edge_index' in data
        has_spatial = 'edge_feat_dist' in data and 'edge_index' in data
        
        if not (has_chem or has_spatial):
            raise ValueError("Molecule data must contain either chemical edges or spatial edges")
        
        # Embed and concatenate edges
        edge_indices = []
        edge_embs = []
        
        if has_chem:
            # Embed chemical edge features [E_chem, 3]
            chem_feat = data['chem_edge_feat_cat']
            offset_feats = chem_feat.long() + self.mol_chem_offset.unsqueeze(0)
            embeddings = self.mol_chem_embed(offset_feats)  # [E_chem, 3, hidden]
            h = embeddings.flatten(1)
            chem_edge_emb = self.mol_chem_combine(h)
            
            edge_indices.append(data['chem_edge_index'])
            edge_embs.append(chem_edge_emb)
        
        if has_spatial:
            # Embed spatial edge features (distances)
            rbf_feats = self.rbf(data['edge_feat_dist'])  # [E_spatial, num_rbf]
            spatial_edge_emb = self.dist_proj(rbf_feats)
            
            edge_indices.append(data['edge_index'])
            edge_embs.append(spatial_edge_emb)
        
        # Concatenate all edge types (no offset needed - same node set)
        edge_index = torch.cat(edge_indices, dim=1)  # [2, E_total]
        edge_emb = torch.cat(edge_embs, dim=0)  # [E_total, hidden_dim]
        
        return {
            'node_emb': node_emb,
            'edge_emb': edge_emb,
            'edge_index': edge_index,
            'pos': data['pos'],
        }
