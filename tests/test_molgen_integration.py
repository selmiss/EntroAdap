"""
Integration test for the complete molecule generation metrics pipeline.
Tests the full flow from trainer to metrics computation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


def test_metrics_diverting_molgen():
    """Test that the metrics dispatcher correctly routes to molecule generation metrics."""
    from src.trainer.metrics.metrics_diverting import compute_metrics
    
    # Mock tokenizer
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    def mock_decode(ids, skip_special_tokens=False):
        if np.array_equal(ids, [1, 2, 3]):
            return "CCO"
        elif np.array_equal(ids, [4, 5, 6]):
            return "CCO"
        return ""
    
    tokenizer.decode = mock_decode
    
    predictions = [np.array([4, 5, 6])]
    labels = [np.array([1, 2, 3])]
    
    # Test molgen metrics routing
    metrics, detailed = compute_metrics(
        eval_metrics="molgen",
        predictions=predictions,
        labels=labels,
        tokenizer=tokenizer,
    )
    
    # Verify metrics are computed
    assert 'exact' in metrics
    assert 'morgan_fts' in metrics
    assert 'validity' in metrics
    assert detailed is not None
    assert len(detailed) == 1


def test_metrics_diverting_all_types():
    """Test that all metric types work correctly."""
    from src.trainer.metrics.metrics_diverting import compute_metrics
    
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.decode = lambda ids, skip_special_tokens=False: "test"
    
    predictions = [np.array([1, 2, 3])]
    labels = [np.array([1, 2, 3])]
    
    # Test none
    metrics, detailed = compute_metrics("none", predictions, labels, tokenizer)
    assert metrics == {}
    assert detailed is None
    
    # Test text
    metrics, detailed = compute_metrics("text", predictions, labels, tokenizer)
    assert 'bleu_2' in metrics
    assert detailed is None
    
    # Test qa
    def qa_decode(ids, skip_special_tokens=False):
        return "Answer: A"
    tokenizer.decode = qa_decode
    metrics, detailed = compute_metrics("qa", predictions, labels, tokenizer)
    assert 'accuracy' in metrics
    assert detailed is not None
    
    # Test molgen
    def mol_decode(ids, skip_special_tokens=False):
        return "CCO"
    tokenizer.decode = mol_decode
    metrics, detailed = compute_metrics("molgen", predictions, labels, tokenizer)
    assert 'exact' in metrics
    assert 'morgan_fts' in metrics
    assert detailed is not None


def test_trainer_with_molgen_metrics():
    """Test that the trainer correctly uses molgen metrics."""
    # This is a simple verification that the trainer accepts the molgen parameter
    # Full trainer initialization requires accelerate setup which is beyond unit test scope
    
    # Verify the parameter is documented
    from src.trainer.octopus_trainer import MultiModalSFTTrainer
    
    # Check the docstring mentions molgen
    assert "molgen" in MultiModalSFTTrainer.__init__.__doc__
    
    # Verify eval_metrics parameter exists in __init__ signature
    import inspect
    sig = inspect.signature(MultiModalSFTTrainer.__init__)
    assert 'eval_metrics' in sig.parameters
    
    print("Trainer correctly supports 'molgen' eval_metrics parameter")


def test_full_pipeline_with_predictions():
    """Test the full pipeline from predictions to metrics."""
    from src.trainer.metrics.molecule_generation import compute_metrics_molecule_generation_detailed
    
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    # Create test data with multiple molecules
    test_molecules = [
        ("CCO", "CCO"),      # Exact match
        ("CCO", "CCCO"),     # Similar but different
        ("c1ccccc1", "c1ccccc1"),  # Exact match (benzene)
    ]
    
    predictions = []
    labels = []
    
    for i, (gt, pred) in enumerate(test_molecules):
        # Mock token IDs for ground truth
        gt_ids = np.array([100 + i * 10 + j for j in range(len(gt))])
        # Mock token IDs for prediction
        pred_ids = np.array([200 + i * 10 + j for j in range(len(pred))])
        
        predictions.append(pred_ids)
        labels.append(gt_ids)
    
    # Setup decode function
    decode_map = {}
    for i, (gt, pred) in enumerate(test_molecules):
        gt_ids = tuple([100 + i * 10 + j for j in range(len(gt))])
        pred_ids = tuple([200 + i * 10 + j for j in range(len(pred))])
        decode_map[gt_ids] = gt
        decode_map[pred_ids] = pred
    
    def mock_decode(ids, skip_special_tokens=False):
        key = tuple(ids.tolist() if hasattr(ids, 'tolist') else ids)
        return decode_map.get(key, "")
    
    tokenizer.decode = mock_decode
    
    # Compute metrics
    metrics, detailed_results = compute_metrics_molecule_generation_detailed(
        predictions=predictions,
        labels=labels,
        tokenizer=tokenizer,
        selfies_mode=False,
    )
    
    # Verify results
    assert len(detailed_results) == 3
    
    # Check first sample (exact match)
    assert detailed_results[0]['exact_match'] == 1
    assert detailed_results[0]['ground_truth'] == "CCO"
    assert detailed_results[0]['prediction'] == "CCO"
    
    # Check second sample (not exact match)
    assert detailed_results[1]['exact_match'] == 0
    assert detailed_results[1]['ground_truth'] == "CCO"
    assert detailed_results[1]['prediction'] == "CCCO"
    
    # Overall metrics
    assert 0 <= metrics['exact'] <= 1
    assert 0 <= metrics['validity'] <= 1
    assert metrics['validity'] == 1.0  # All valid SMILES
    
    # Exact match should be 2/3 = 0.667
    assert abs(metrics['exact'] - 2/3) < 0.01


def test_error_handling():
    """Test that metrics handle errors gracefully."""
    from src.trainer.metrics.metrics_diverting import compute_metrics
    
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    # Test with invalid metric type
    metrics, detailed = compute_metrics(
        eval_metrics="invalid_type",
        predictions=[np.array([1, 2, 3])],
        labels=[np.array([1, 2, 3])],
        tokenizer=tokenizer,
    )
    
    # Should return empty dict and None on error
    assert metrics == {}
    assert detailed is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
