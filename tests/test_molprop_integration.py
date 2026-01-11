"""Integration tests for molecular property prediction metrics."""

import pytest
import numpy as np
from unittest.mock import Mock


def test_metrics_diverting_molprop():
    """Test that the metrics dispatcher correctly routes to mol property metrics."""
    from src.trainer.metrics.metrics_diverting import compute_metrics
    
    # Mock tokenizer
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    def mock_decode(ids, skip_special_tokens=False):
        if np.array_equal(ids, [1, 2, 3]):
            return "42.5"
        elif np.array_equal(ids, [4, 5, 6]):
            return "42.0"
        return ""
    
    tokenizer.decode = mock_decode
    
    predictions = [np.array([4, 5, 6])]
    labels = [np.array([1, 2, 3])]
    
    # Test molprop metrics routing
    metrics, detailed = compute_metrics(
        eval_metrics="molprop",
        predictions=predictions,
        labels=labels,
        tokenizer=tokenizer,
    )
    
    # Verify metrics are computed
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'pearson' in metrics
    assert 'valid_ratio' in metrics
    assert detailed is not None
    assert len(detailed) == 1


def test_metrics_diverting_all_types():
    """Test that all metric types including molprop work correctly."""
    from src.trainer.metrics.metrics_diverting import compute_metrics
    
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    predictions = [np.array([1, 2, 3])]
    labels = [np.array([1, 2, 3])]
    
    # Test none
    tokenizer.decode = lambda ids, skip_special_tokens=False: "test"
    metrics, detailed = compute_metrics("none", predictions, labels, tokenizer)
    assert metrics == {}
    assert detailed is None
    
    # Test molprop
    tokenizer.decode = lambda ids, skip_special_tokens=False: "10.5"
    metrics, detailed = compute_metrics("molprop", predictions, labels, tokenizer)
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert detailed is not None


def test_trainer_supports_molprop():
    """Test that the trainer documentation includes molprop."""
    from src.trainer.octopus_trainer import MultiModalSFTTrainer
    import inspect
    
    # Check the docstring mentions molprop
    assert "molprop" in MultiModalSFTTrainer.__init__.__doc__
    
    # Verify eval_metrics parameter exists
    sig = inspect.signature(MultiModalSFTTrainer.__init__)
    assert 'eval_metrics' in sig.parameters
    
    print("Trainer correctly supports 'molprop' eval_metrics parameter")


def test_full_pipeline_with_property_predictions():
    """Test the full pipeline from predictions to metrics for property prediction."""
    from src.trainer.metrics.mol_property import compute_metrics_mol_property_detailed
    
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    # Create test data with multiple property values
    test_properties = [
        ("10.5", "10.2"),    # Close prediction
        ("42.0", "41.5"),    # Close prediction
        ("-3.14", "-3.0"),   # Close prediction with negative
    ]
    
    predictions = []
    labels = []
    
    for i, (gt, pred) in enumerate(test_properties):
        gt_ids = np.array([100 + i * 10 + j for j in range(len(gt))])
        pred_ids = np.array([200 + i * 10 + j for j in range(len(pred))])
        predictions.append(pred_ids)
        labels.append(gt_ids)
    
    # Setup decode function
    decode_map = {}
    for i, (gt, pred) in enumerate(test_properties):
        gt_ids = tuple([100 + i * 10 + j for j in range(len(gt))])
        pred_ids = tuple([200 + i * 10 + j for j in range(len(pred))])
        decode_map[gt_ids] = gt
        decode_map[pred_ids] = pred
    
    def mock_decode(ids, skip_special_tokens=False):
        key = tuple(ids.tolist() if hasattr(ids, 'tolist') else ids)
        return decode_map.get(key, "")
    
    tokenizer.decode = mock_decode
    
    # Compute metrics
    metrics, detailed_results = compute_metrics_mol_property_detailed(
        predictions=predictions,
        labels=labels,
        tokenizer=tokenizer,
    )
    
    # Verify results
    assert len(detailed_results) == 3
    
    # Check first sample
    assert detailed_results[0]['ground_truth_value'] == 10.5
    assert detailed_results[0]['prediction_value'] == 10.2
    assert abs(detailed_results[0]['absolute_error'] - 0.3) < 0.01
    
    # Overall metrics
    assert 0 <= metrics['mae'] < 1.0  # Should be small error
    assert metrics['valid_ratio'] == 1.0  # All valid
    assert -1 <= metrics['r2'] <= 1  # Valid R² range
    assert -1 <= metrics['pearson'] <= 1  # Valid correlation range


def test_property_with_scientific_notation():
    """Test handling of scientific notation in property predictions."""
    from src.trainer.metrics.mol_property import compute_metrics_mol_property_detailed
    
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    test_data = [
        ("1.5e-3", "1.6e-3"),
        ("2.3E+5", "2.2E+5"),
    ]
    
    predictions = []
    labels = []
    decode_map = {}
    
    for i, (gt, pred) in enumerate(test_data):
        gt_ids = tuple([100 + i * 10 + j for j in range(5)])
        pred_ids = tuple([200 + i * 10 + j for j in range(5)])
        decode_map[gt_ids] = gt
        decode_map[pred_ids] = pred
        predictions.append(np.array(pred_ids))  # predictions use pred_ids
        labels.append(np.array(gt_ids))         # labels use gt_ids
    
    tokenizer.decode = lambda ids, skip_special_tokens=False: decode_map.get(tuple(ids.tolist()), "")
    
    metrics, detailed = compute_metrics_mol_property_detailed(predictions, labels, tokenizer)
    
    # Should handle scientific notation correctly
    assert metrics['valid_ratio'] == 1.0
    # Predictions are 1.6e-3 and 2.2E+5
    assert abs(detailed[0]['prediction_value'] - 1.6e-3) < 1e-6
    assert abs(detailed[1]['prediction_value'] - 2.2e5) < 1.0
    # Ground truth are 1.5e-3 and 2.3E+5
    assert abs(detailed[0]['ground_truth_value'] - 1.5e-3) < 1e-6
    assert abs(detailed[1]['ground_truth_value'] - 2.3e5) < 1.0


def test_property_with_units():
    """Test extraction of numbers with units."""
    from src.trainer.metrics.mol_property import compute_metrics_mol_property_detailed
    
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    test_data = [
        ("42.5 kcal/mol", "42.0"),
        ("-3.14 eV", "-3.0 eV"),
    ]
    
    predictions = []
    labels = []
    decode_map = {}
    
    for i, (gt, pred) in enumerate(test_data):
        gt_ids = tuple([100 + i * 10 + j for j in range(5)])
        pred_ids = tuple([200 + i * 10 + j for j in range(5)])
        decode_map[gt_ids] = gt
        decode_map[pred_ids] = pred
        predictions.append(np.array(pred_ids))
        labels.append(np.array(gt_ids))
    
    tokenizer.decode = lambda ids, skip_special_tokens=False: decode_map.get(tuple(ids.tolist()), "")
    
    metrics, detailed = compute_metrics_mol_property_detailed(predictions, labels, tokenizer)
    
    # Should extract numbers correctly ignoring units
    assert metrics['valid_ratio'] == 1.0
    assert detailed[0]['ground_truth_value'] == 42.5
    assert detailed[0]['prediction_value'] == 42.0
    assert detailed[1]['ground_truth_value'] == -3.14
    assert detailed[1]['prediction_value'] == -3.0


def test_error_handling_molprop():
    """Test that metrics handle errors gracefully."""
    from src.trainer.metrics.metrics_diverting import compute_metrics
    
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.decode = lambda ids, skip_special_tokens=False: "10.5"
    
    # Test with empty predictions
    metrics, detailed = compute_metrics(
        eval_metrics="molprop",
        predictions=[],
        labels=[],
        tokenizer=tokenizer,
    )
    
    # Should return empty results gracefully
    assert metrics == {}
    assert detailed is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
