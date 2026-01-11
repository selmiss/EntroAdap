"""Tests for molecule generation metrics."""

import pytest
import numpy as np
from unittest.mock import Mock


def test_compute_metrics_molecule_generation():
    """Test basic molecule generation metrics computation."""
    from src.trainer.metrics.molecule_generation import compute_metrics_molecule_generation
    
    # Mock tokenizer
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    # Create sample data - SMILES strings
    # Using simple valid SMILES
    gt_smiles = ["CCO", "c1ccccc1"]  # Ethanol, Benzene
    pred_smiles = ["CCO", "c1ccccc1"]  # Perfect matches
    
    def mock_decode(ids, skip_special_tokens=False):
        # Map token IDs back to SMILES strings
        if np.array_equal(ids, [1, 2, 3]):
            return gt_smiles[0]
        elif np.array_equal(ids, [4, 5, 6, 7, 8, 9, 10]):
            return gt_smiles[1]
        elif np.array_equal(ids, [11, 12, 13]):
            return pred_smiles[0]
        elif np.array_equal(ids, [14, 15, 16, 17, 18, 19, 20]):
            return pred_smiles[1]
        return ""
    
    tokenizer.decode = mock_decode
    
    # Create mock predictions and labels (token IDs)
    predictions = [
        np.array([11, 12, 13]),
        np.array([14, 15, 16, 17, 18, 19, 20]),
    ]
    labels = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6, 7, 8, 9, 10]),
    ]
    
    # Compute metrics (SMILES mode, not SELFIES)
    metrics = compute_metrics_molecule_generation(predictions, labels, tokenizer, selfies_mode=False)
    
    # Verify metrics structure
    assert 'exact' in metrics
    assert 'bleu' in metrics
    assert 'levenshtein' in metrics
    assert 'rdk_fts' in metrics
    assert 'maccs_fts' in metrics
    assert 'morgan_fts' in metrics
    assert 'validity' in metrics
    
    # With perfect matches, exact should be 1.0
    assert metrics['exact'] == 1.0
    # Validity should be 1.0 (all valid SMILES)
    assert metrics['validity'] == 1.0
    # Fingerprint similarities should be 1.0 for identical molecules
    assert metrics['rdk_fts'] == 1.0
    assert metrics['maccs_fts'] == 1.0
    assert metrics['morgan_fts'] == 1.0
    # Levenshtein distance should be 0 for exact matches
    assert metrics['levenshtein'] == 0.0


def test_compute_metrics_molecule_generation_detailed():
    """Test detailed molecule generation metrics computation."""
    from src.trainer.metrics.molecule_generation import compute_metrics_molecule_generation_detailed
    
    # Mock tokenizer
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    # Create sample data
    gt_smiles = ["CCO"]
    pred_smiles = ["CC"]  # Close but not exact
    
    def mock_decode(ids, skip_special_tokens=False):
        if np.array_equal(ids, [1, 2, 3]):
            return gt_smiles[0]
        elif np.array_equal(ids, [4, 5]):
            return pred_smiles[0]
        elif np.array_equal(ids, [100, 101]):
            return "prompt text"
        return ""
    
    tokenizer.decode = mock_decode
    
    predictions = [np.array([4, 5])]
    labels = [np.array([1, 2, 3])]
    prompts = [np.array([100, 101])]
    
    # Compute metrics with prompts
    metrics, detailed_results = compute_metrics_molecule_generation_detailed(
        predictions, labels, tokenizer, selfies_mode=False, prompts=prompts
    )
    
    # Verify structure
    assert isinstance(metrics, dict)
    assert isinstance(detailed_results, list)
    assert len(detailed_results) == 1
    
    # Verify detailed results contain expected fields
    result = detailed_results[0]
    assert 'index' in result
    assert 'ground_truth' in result
    assert 'prediction' in result
    assert 'ground_truth_smiles' in result
    assert 'prediction_smiles' in result
    assert 'exact_match' in result
    assert 'prompt' in result
    
    # With non-exact match, exact_match should be 0
    assert result['exact_match'] == 0
    assert result['prompt'] == "prompt text"


def test_fingerprint_similarity():
    """Test fingerprint similarity computation."""
    from src.trainer.metrics.molecule_generation import _compute_fingerprint_similarity
    
    # Test identical molecules
    sim = _compute_fingerprint_similarity("CCO", "CCO", fp_type='morgan')
    assert sim == 1.0
    
    # Test different molecules (should be less than 1.0)
    sim = _compute_fingerprint_similarity("CCO", "CCCO", fp_type='morgan')
    assert 0.0 < sim < 1.0
    
    # Test invalid SMILES
    sim = _compute_fingerprint_similarity("invalid", "CCO", fp_type='morgan')
    assert sim == 0.0
    
    # Test different fingerprint types
    for fp_type in ['morgan', 'rdk', 'maccs']:
        sim = _compute_fingerprint_similarity("CCO", "CCO", fp_type=fp_type)
        assert sim == 1.0


def test_levenshtein_distance():
    """Test Levenshtein distance computation."""
    from src.trainer.metrics.molecule_generation import _levenshtein_distance
    
    # Test identical strings
    assert _levenshtein_distance("hello", "hello") == 0
    
    # Test single substitution
    assert _levenshtein_distance("hello", "hallo") == 1
    
    # Test single insertion
    assert _levenshtein_distance("hello", "helllo") == 1
    
    # Test single deletion
    assert _levenshtein_distance("hello", "helo") == 1
    
    # Test empty strings
    assert _levenshtein_distance("", "") == 0
    assert _levenshtein_distance("hello", "") == 5


def test_metrics_with_empty_outputs():
    """Test metrics computation with empty outputs."""
    from src.trainer.metrics.molecule_generation import _compute_metrics_reaction_internal
    
    metrics, per_sample = _compute_metrics_reaction_internal([], selfies_mode=False)
    
    # Should return default values for empty inputs
    assert metrics['exact'] == 0.0
    assert metrics['bleu'] == 0.0
    assert metrics['levenshtein'] == float('inf')
    assert metrics['validity'] == 0.0
    assert len(per_sample) == 0


def test_metrics_with_invalid_smiles():
    """Test metrics computation with invalid SMILES."""
    from src.trainer.metrics.molecule_generation import _compute_metrics_reaction_internal
    
    outputs = [
        {'ground_truth': 'CCO', 'prediction': 'invalid_smiles'}
    ]
    
    metrics, per_sample = _compute_metrics_reaction_internal(outputs, selfies_mode=False)
    
    # Should handle invalid SMILES gracefully
    assert 'validity' in metrics
    assert metrics['validity'] == 0.0  # 0 out of 1 valid
    assert per_sample[0]['prediction_smiles'] is None
    assert per_sample[0]['rdk_fts'] == 0.0
    assert per_sample[0]['maccs_fts'] == 0.0
    assert per_sample[0]['morgan_fts'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
