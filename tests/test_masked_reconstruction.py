"""
Tests for masked reconstruction trainer
"""

import pytest
import torch
import torch.nn as nn

from src.models.aa_encoder import AAEncoder
from src.trainer.reconstruction import ReconstructionTrainer


def create_protein_data(N, E):
    """Create valid protein graph data."""
    node_feat = torch.cat([
        torch.randint(0, 119, (N, 1)),  # atomic_num
        torch.randint(0, 46, (N, 1)),   # atom_name
        torch.randint(0, 24, (N, 1)),   # residue
        torch.randint(0, 27, (N, 1)),   # chain
        torch.randint(1, 100, (N, 1)),  # residue_id
        torch.randint(0, 2, (N, 1)),    # is_backbone
        torch.randint(0, 2, (N, 1)),    # is_ca
    ], dim=1)
    
    return {
        'modality': 'protein',
        'value': {
            'node_feat': node_feat,
            'edge_attr': torch.rand(E, 1) * 10,
            'edge_index': torch.randint(0, N, (2, E)),
            'pos': torch.randn(N, 3),
        }
    }


def create_molecule_data(N, E_chem=None, E_spatial=None):
    """Create valid molecule graph data."""
    node_feat = torch.cat([
        torch.randint(0, 119, (N, 1)),  # atomic_num
        torch.randint(0, 4, (N, 1)),    # chirality
        torch.randint(0, 12, (N, 1)),   # degree
        torch.randint(0, 12, (N, 1)),   # charge
        torch.randint(0, 10, (N, 1)),   # numH
        torch.randint(0, 6, (N, 1)),    # radical
        torch.randint(0, 6, (N, 1)),    # hybrid
        torch.randint(0, 2, (N, 1)),    # aromatic
        torch.randint(0, 2, (N, 1)),    # ring
    ], dim=1)
    
    value = {
        'node_feat': node_feat,
        'pos': torch.randn(N, 3),
    }
    
    if E_chem is not None:
        chem_edge_feat = torch.cat([
            torch.randint(0, 5, (E_chem, 1)),
            torch.randint(0, 6, (E_chem, 1)),
            torch.randint(0, 2, (E_chem, 1)),
        ], dim=1)
        value['chem_edge_index'] = torch.randint(0, N, (2, E_chem))
        value['chem_edge_feat_cat'] = chem_edge_feat
    
    if E_spatial is not None:
        value['edge_index'] = torch.randint(0, N, (2, E_spatial))
        value['edge_feat_dist'] = torch.rand(E_spatial, 1) * 10
    
    return {
        'modality': 'molecule',
        'value': value
    }


@pytest.fixture
def encoder():
    return AAEncoder(
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
        update_coords=True,
        use_layernorm=True,
        num_rbf=16,
    )


@pytest.fixture
def trainer(encoder):
    return ReconstructionTrainer(
        encoder=encoder,
        num_elements=119,
        num_dist_bins=32,
        dist_min=0.0,
        dist_max=20.0,
        element_weight=1.0,
        dist_weight=1.0,
        noise_weight=1.0,
    )


def test_trainer_initialization(trainer):
    assert trainer.num_elements == 119
    assert trainer.num_dist_bins == 32
    assert trainer.element_weight == 1.0
    assert trainer.dist_weight == 1.0
    assert trainer.noise_weight == 1.0
    assert hasattr(trainer, 'element_head')
    assert hasattr(trainer, 'dist_head')
    assert hasattr(trainer, 'noise_head')


def test_digitize_basic(trainer):
    values = torch.tensor([0.5, 5.0, 10.0, 15.0, 19.5])
    bins = torch.linspace(0.0, 20.0, 33)  # 32 bins
    
    indices = trainer.digitize(values, bins)
    
    assert indices.shape == values.shape
    assert (indices >= 0).all()
    assert (indices < 32).all()


def test_digitize_out_of_range(trainer):
    values = torch.tensor([-5.0, 25.0])
    bins = torch.linspace(0.0, 20.0, 33)
    
    indices = trainer.digitize(values, bins)
    
    # Should be clamped to valid range
    assert (indices >= 0).all()
    assert (indices < 32).all()


def test_digitize_multidim(trainer):
    values = torch.randn(10, 3)
    bins = torch.linspace(-2.0, 2.0, 33)
    
    indices = trainer.digitize(values, bins)
    
    assert indices.shape == values.shape
    assert (indices >= 0).all()
    assert (indices < 32).all()


def test_forward_protein_no_mask(trainer):
    data = create_protein_data(N=20, E=40)
    
    result = trainer(data, compute_loss=False)
    
    assert 'node_emb' in result
    assert 'pos' in result
    assert result['node_emb'].shape == (20, 64)
    assert result['pos'].shape == (20, 3)
    assert 'loss' not in result


def test_forward_molecule_no_mask(trainer):
    data = create_molecule_data(N=15, E_chem=30, E_spatial=25)
    
    result = trainer(data, compute_loss=False)
    
    assert 'node_emb' in result
    assert 'pos' in result
    assert result['node_emb'].shape == (15, 64)
    assert result['pos'].shape == (15, 3)


def test_forward_with_node_mask(trainer):
    N = 20
    E = 40
    
    node_mask = torch.zeros(N, dtype=torch.bool)
    node_mask[:5] = True
    
    element_labels = torch.randint(0, 119, (N,))
    
    data = create_protein_data(N, E)
    
    result = trainer(
        data,
        node_mask=node_mask,
        element_labels=element_labels,
        compute_loss=True
    )
    
    assert 'element_logits_masked' in result
    assert result['element_logits_masked'].shape == (5, 119)
    assert result['num_masked_nodes'] == 5
    assert 'element_loss' in result
    assert result['element_loss'].ndim == 0
    assert 'loss' in result


def test_forward_with_edge_mask(trainer):
    N = 20
    E = 40
    
    edge_mask = torch.zeros(E, dtype=torch.bool)
    edge_mask[:10] = True
    
    dist_labels = torch.rand(E) * 15.0
    
    data = create_protein_data(N, E)
    
    result = trainer(
        data,
        edge_mask=edge_mask,
        dist_labels=dist_labels,
        compute_loss=True
    )
    
    assert 'dist_logits' in result
    assert result['dist_logits'].shape == (10, 32)
    assert result['num_masked_edges'] == 10
    assert 'dist_loss' in result
    assert result['dist_loss'].ndim == 0
    assert 'loss' in result


def test_forward_with_noise_mask(trainer):
    N = 20
    E = 40
    
    node_mask = torch.zeros(N, dtype=torch.bool)
    node_mask[:5] = True
    
    noise_labels = torch.randn(N, 3) * 0.5
    
    data = create_protein_data(N, E)
    
    result = trainer(
        data,
        node_mask=node_mask,
        noise_labels=noise_labels,
        compute_loss=True
    )
    
    assert 'noise_pred' in result
    assert result['noise_pred'].shape == (N, 3)
    assert 'noise_loss' in result
    assert result['noise_loss'].ndim == 0
    assert 'loss' in result


def test_forward_all_masks(trainer):
    N = 20
    E = 40
    
    node_mask = torch.zeros(N, dtype=torch.bool)
    node_mask[:5] = True
    
    edge_mask = torch.zeros(E, dtype=torch.bool)
    edge_mask[:10] = True
    
    element_labels = torch.randint(0, 119, (N,))
    dist_labels = torch.rand(E) * 15.0
    noise_labels = torch.randn(N, 3) * 0.5
    
    data = create_protein_data(N, E)
    
    result = trainer(
        data,
        node_mask=node_mask,
        edge_mask=edge_mask,
        element_labels=element_labels,
        dist_labels=dist_labels,
        noise_labels=noise_labels,
        compute_loss=True
    )
    
    # Check all outputs
    assert 'node_emb' in result
    assert 'pos' in result
    assert 'element_logits_masked' in result
    assert 'dist_logits' in result
    assert 'noise_pred' in result
    assert result['num_masked_nodes'] == 5
    assert result['num_masked_edges'] == 10
    
    # Check all losses
    assert 'element_loss' in result
    assert 'dist_loss' in result
    assert 'noise_loss' in result
    assert 'loss' in result
    
    # Loss should be weighted sum
    expected_loss = result['element_loss'] + result['dist_loss'] + result['noise_loss']
    assert torch.allclose(result['loss'], expected_loss, rtol=1e-5)


def test_backward_pass(trainer):
    N = 20
    E = 40
    
    node_mask = torch.zeros(N, dtype=torch.bool)
    node_mask[:5] = True
    
    element_labels = torch.randint(0, 119, (N,))
    
    data = create_protein_data(N, E)
    
    result = trainer(
        data,
        node_mask=node_mask,
        element_labels=element_labels,
        compute_loss=True
    )
    
    loss = result['loss']
    loss.backward()
    
    # Check that at least some parameters have gradients
    has_grad = sum(1 for p in trainer.parameters() if p.requires_grad and p.grad is not None)
    total_params = sum(1 for p in trainer.parameters() if p.requires_grad)
    assert has_grad > 0, "No gradients computed"
    assert has_grad > total_params * 0.5, "Too few parameters have gradients"


def test_batch_processing(trainer):
    N = 25
    E = 50
    
    batch = torch.tensor([0]*10 + [1]*15)
    
    data = create_protein_data(N, E)
    
    result = trainer(data, batch=batch, compute_loss=False)
    
    assert result['node_emb'].shape == (N, 64)
    assert result['pos'].shape == (N, 3)


def test_element_prediction_distribution(trainer):
    N = 20
    E = 40
    
    node_mask = torch.ones(N, dtype=torch.bool)
    element_labels = torch.randint(0, 119, (N,))
    
    data = create_protein_data(N, E)
    
    result = trainer(
        data,
        node_mask=node_mask,
        element_labels=element_labels,
        compute_loss=True
    )
    
    logits = result['element_logits_masked']
    probs = torch.softmax(logits, dim=-1)
    
    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(N), rtol=1e-5)
    
    # Check all probabilities are positive
    assert (probs >= 0).all()


def test_distance_binning_edges(trainer):
    # Test edge cases for distance binning
    dist_labels = torch.tensor([0.0, 10.0, 20.0, -1.0, 25.0])
    bins = trainer.dist_bins
    
    indices = trainer.digitize(dist_labels, bins)
    
    # First bin for 0.0
    assert indices[0] == 0
    
    # Middle bin for 10.0
    assert 0 < indices[1] < 31
    
    # Last bin for 20.0
    assert indices[2] == 31
    
    # Out of range values should be clamped
    assert indices[3] == 0
    assert indices[4] == 31


def test_soft_targets(trainer):
    # Test soft target generation for distance bins
    distances = torch.tensor([5.0, 10.0, 15.0])
    soft_targets = trainer.distances_to_soft_targets(distances, sigma=0.5)
    
    assert soft_targets.shape == (3, 32)
    assert torch.allclose(soft_targets.sum(dim=-1), torch.ones(3), rtol=1e-5)
    assert (soft_targets >= 0).all()


def test_noise_binning_edges(trainer):
    # Noise is now direct regression, test prediction shape
    N = 10
    E = 20
    
    node_mask = torch.zeros(N, dtype=torch.bool)
    node_mask[:5] = True
    
    noise_labels = torch.randn(N, 3) * 0.5
    
    data = create_protein_data(N, E)
    result = trainer(data, node_mask=node_mask, noise_labels=noise_labels, compute_loss=True)
    
    assert result['noise_pred'].shape == (N, 3)
    assert 'noise_loss' in result


def test_no_loss_without_labels(trainer):
    N = 20
    E = 40
    
    node_mask = torch.zeros(N, dtype=torch.bool)
    node_mask[:5] = True
    
    data = create_protein_data(N, E)
    
    result = trainer(data, node_mask=node_mask, compute_loss=True)
    
    assert 'element_logits_masked' in result
    assert 'element_loss' not in result
    # Loss is 0.0 when no supervised tasks are computed
    assert result['loss'] == 0.0


def test_distance_head_masked_only(trainer):
    # Test that distance head only computes for masked edges
    N = 20
    E = 40
    
    edge_mask = torch.zeros(E, dtype=torch.bool)
    edge_mask[:10] = True  # Only 10 edges masked
    
    dist_labels = torch.rand(E) * 15.0
    
    data = create_protein_data(N, E)
    
    result = trainer(
        data,
        edge_mask=edge_mask,
        dist_labels=dist_labels,
        compute_loss=True
    )
    
    # Should only compute logits for masked edges
    assert result['dist_logits'].shape == (10, 32)
    assert 'dist_loss' in result


def test_soft_distance_targets(trainer):
    # Test distance prediction with soft targets
    N = 20
    E = 40
    
    edge_mask = torch.zeros(E, dtype=torch.bool)
    edge_mask[:10] = True
    
    # Soft targets: probability distributions over bins
    soft_targets = torch.softmax(torch.randn(E, 32), dim=-1)
    
    data = create_protein_data(N, E)
    
    result = trainer(
        data,
        edge_mask=edge_mask,
        dist_labels=soft_targets,
        compute_loss=True
    )
    
    assert 'dist_logits' in result
    assert 'dist_loss' in result


def test_loss_weights(trainer):
    # Test weighted loss composition
    N = 20
    E = 40
    
    # Create trainer with custom weights
    encoder = AAEncoder(hidden_dim=64, num_layers=2)
    weighted_trainer = ReconstructionTrainer(
        encoder=encoder,
        element_weight=2.0,
        dist_weight=1.5,
        noise_weight=0.5,
    )
    
    node_mask = torch.zeros(N, dtype=torch.bool)
    node_mask[:5] = True
    edge_mask = torch.zeros(E, dtype=torch.bool)
    edge_mask[:10] = True
    
    element_labels = torch.randint(0, 119, (N,))
    dist_labels = torch.rand(E) * 15.0
    noise_labels = torch.randn(N, 3) * 0.5
    
    data = create_protein_data(N, E)
    
    result = weighted_trainer(
        data,
        node_mask=node_mask,
        edge_mask=edge_mask,
        element_labels=element_labels,
        dist_labels=dist_labels,
        noise_labels=noise_labels,
        compute_loss=True
    )
    
    expected_loss = (
        2.0 * result['element_loss'] +
        1.5 * result['dist_loss'] +
        0.5 * result['noise_loss']
    )
    assert torch.allclose(result['loss'], expected_loss, rtol=1e-5)


def test_loss_normalization_invariance(trainer):
    # Test that loss magnitude doesn't change with different mask sizes
    N = 20
    E = 40
    
    data = create_protein_data(N, E)
    element_labels = torch.randint(0, 119, (N,))
    
    # Small mask
    small_mask = torch.zeros(N, dtype=torch.bool)
    small_mask[:2] = True
    
    result_small = trainer(
        data,
        node_mask=small_mask,
        element_labels=element_labels,
        compute_loss=True
    )
    
    # Large mask
    large_mask = torch.zeros(N, dtype=torch.bool)
    large_mask[:10] = True
    
    result_large = trainer(
        data,
        node_mask=large_mask,
        element_labels=element_labels,
        compute_loss=True
    )
    
    # Both losses should be per-item, so comparable in magnitude
    # (Not exactly equal due to different samples, but order of magnitude similar)
    assert 0.1 < result_small['element_loss'] / result_large['element_loss'] < 10.0


def test_molecule_with_only_chem_edges(trainer):
    data = create_molecule_data(N=15, E_chem=30, E_spatial=None)
    
    result = trainer(data, compute_loss=False)
    
    assert result['node_emb'].shape == (15, 64)
    assert result['pos'].shape == (15, 3)


def test_molecule_with_only_spatial_edges(trainer):
    data = create_molecule_data(N=15, E_chem=None, E_spatial=25)
    
    result = trainer(data, compute_loss=False)
    
    assert result['node_emb'].shape == (15, 64)
    assert result['pos'].shape == (15, 3)
