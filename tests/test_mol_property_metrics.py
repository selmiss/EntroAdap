"""Tests for molecular property prediction metrics."""

import pytest
import numpy as np
from unittest.mock import Mock


def test_compute_metrics_mol_property():
    """Test basic molecular property prediction metrics computation."""
    from src.trainer.metrics.mol_property import compute_metrics_mol_property
    
    # Mock tokenizer
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    # Create sample data - property values
    gt_values = ["42.5", "-3.14", "0.0"]
    pred_values = ["42.0", "-3.0", "0.5"]
    
    def mock_decode(ids, skip_special_tokens=False):
        # Map token IDs to values
        if np.array_equal(ids, [1, 2, 3]):
            return gt_values[0]
        elif np.array_equal(ids, [4, 5, 6]):
            return gt_values[1]
        elif np.array_equal(ids, [7, 8, 9]):
            return gt_values[2]
        elif np.array_equal(ids, [11, 12, 13]):
            return pred_values[0]
        elif np.array_equal(ids, [14, 15, 16]):
            return pred_values[1]
        elif np.array_equal(ids, [17, 18, 19]):
            return pred_values[2]
        return ""
    
    tokenizer.decode = mock_decode
    
    # Create mock predictions and labels (token IDs)
    predictions = [
        np.array([11, 12, 13]),
        np.array([14, 15, 16]),
        np.array([17, 18, 19]),
    ]
    labels = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9]),
    ]
    
    # Compute metrics
    metrics = compute_metrics_mol_property(predictions, labels, tokenizer)
    
    # Verify metrics structure
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'pearson' in metrics
    assert 'valid_ratio' in metrics
    
    # All values should be valid
    assert metrics['valid_ratio'] == 1.0
    
    # MAE should be positive
    assert metrics['mae'] > 0
    
    # RMSE should be >= MAE
    assert metrics['rmse'] >= metrics['mae']


def test_compute_metrics_mol_property_detailed():
    """Test detailed molecular property prediction metrics computation."""
    from src.trainer.metrics.mol_property import compute_metrics_mol_property_detailed
    
    # Mock tokenizer
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    # Create sample data
    gt_values = ["10.5"]
    pred_values = ["12.0"]
    
    def mock_decode(ids, skip_special_tokens=False):
        if np.array_equal(ids, [1, 2, 3]):
            return gt_values[0]
        elif np.array_equal(ids, [4, 5, 6]):
            return pred_values[0]
        elif np.array_equal(ids, [100, 101]):
            return "What is the property value?"
        return ""
    
    tokenizer.decode = mock_decode
    
    predictions = [np.array([4, 5, 6])]
    labels = [np.array([1, 2, 3])]
    prompts = [np.array([100, 101])]
    
    # Compute metrics with prompts
    metrics, detailed_results = compute_metrics_mol_property_detailed(
        predictions, labels, tokenizer, prompts=prompts
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
    assert 'ground_truth_value' in result
    assert 'prediction_value' in result
    assert 'absolute_error' in result
    assert 'squared_error' in result
    assert 'valid' in result
    assert 'prompt' in result
    
    # Check values
    assert result['ground_truth_value'] == 10.5
    assert result['prediction_value'] == 12.0
    assert result['absolute_error'] == 1.5
    assert result['valid'] is True
    assert result['prompt'] == "What is the property value?"


def test_extract_number():
    """Test number extraction from text."""
    from src.trainer.metrics.mol_property import _extract_number
    
    # Test plain numbers
    assert _extract_number("42") == 42.0
    assert _extract_number("-3.14") == -3.14
    assert _extract_number("0.5") == 0.5
    assert _extract_number("+2.5") == 2.5
    
    # Test scientific notation
    assert _extract_number("1.5e-3") == 1.5e-3
    assert _extract_number("2.3E+5") == 2.3e5
    assert _extract_number("-4.2e-2") == -4.2e-2
    
    # Test with units
    assert _extract_number("42.5 kcal/mol") == 42.5
    assert _extract_number("-3.14 eV") == -3.14
    assert _extract_number("0.5 Å") == 0.5
    
    # Test in sentences
    assert _extract_number("The value is 42.5") == 42.5
    assert _extract_number("Result: -3.14") == -3.14
    
    # Test edge cases
    assert _extract_number("") is None
    assert _extract_number("no number here") is None
    assert _extract_number(None) is None


def test_metrics_with_perfect_predictions():
    """Test metrics with perfect predictions (MAE=0)."""
    from src.trainer.metrics.mol_property import _compute_metrics_property_internal
    
    outputs = [
        {'ground_truth': '10.0', 'prediction': '10.0'},
        {'ground_truth': '20.0', 'prediction': '20.0'},
        {'ground_truth': '30.0', 'prediction': '30.0'},
    ]
    
    metrics, per_sample = _compute_metrics_property_internal(outputs)
    
    # Perfect predictions should have MAE=0
    assert metrics['mae'] == 0.0
    assert metrics['mse'] == 0.0
    assert metrics['rmse'] == 0.0
    assert metrics['r2'] == 1.0  # Perfect R²
    assert metrics['pearson'] == 1.0  # Perfect correlation
    assert metrics['valid_ratio'] == 1.0


def test_metrics_with_negative_numbers():
    """Test metrics with negative numbers."""
    from src.trainer.metrics.mol_property import _compute_metrics_property_internal
    
    outputs = [
        {'ground_truth': '-10.0', 'prediction': '-9.0'},
        {'ground_truth': '-20.0', 'prediction': '-21.0'},
        {'ground_truth': '-5.5', 'prediction': '-5.0'},
    ]
    
    metrics, per_sample = _compute_metrics_property_internal(outputs)
    
    # Should handle negative numbers correctly
    assert metrics['mae'] > 0
    assert metrics['valid_ratio'] == 1.0
    assert all(sample['valid'] for sample in per_sample)


def test_metrics_with_invalid_predictions():
    """Test metrics with invalid (non-numeric) predictions."""
    from src.trainer.metrics.mol_property import _compute_metrics_property_internal
    
    outputs = [
        {'ground_truth': '10.0', 'prediction': 'invalid'},
        {'ground_truth': '20.0', 'prediction': 'not a number'},
        {'ground_truth': '30.0', 'prediction': '31.0'},  # Only this one valid
    ]
    
    metrics, per_sample = _compute_metrics_property_internal(outputs)
    
    # Should handle invalid predictions gracefully
    assert metrics['valid_ratio'] == 1/3  # Only 1 out of 3 valid
    assert per_sample[0]['valid'] is False
    assert per_sample[1]['valid'] is False
    assert per_sample[2]['valid'] is True


def test_metrics_with_empty_outputs():
    """Test metrics computation with empty outputs."""
    from src.trainer.metrics.mol_property import _compute_metrics_property_internal
    
    metrics, per_sample = _compute_metrics_property_internal([])
    
    # Should return default values for empty inputs
    assert metrics['mae'] == float('inf')
    assert metrics['mse'] == float('inf')
    assert metrics['rmse'] == float('inf')
    assert metrics['r2'] == 0.0
    assert metrics['pearson'] == 0.0
    assert len(per_sample) == 0


def test_mae_calculation():
    """Test specific MAE calculation."""
    from src.trainer.metrics.mol_property import _compute_metrics_property_internal
    
    # Known values with known MAE
    outputs = [
        {'ground_truth': '10.0', 'prediction': '12.0'},  # error = 2.0
        {'ground_truth': '20.0', 'prediction': '18.0'},  # error = 2.0
        {'ground_truth': '30.0', 'prediction': '31.0'},  # error = 1.0
    ]
    
    metrics, per_sample = _compute_metrics_property_internal(outputs)
    
    # Expected MAE = (2.0 + 2.0 + 1.0) / 3 = 1.666...
    expected_mae = 5.0 / 3.0
    assert abs(metrics['mae'] - expected_mae) < 0.001


def test_r2_calculation():
    """Test R² score calculation."""
    from src.trainer.metrics.mol_property import _compute_metrics_property_internal
    
    # Perfect linear relationship
    outputs = [
        {'ground_truth': '1.0', 'prediction': '2.0'},
        {'ground_truth': '2.0', 'prediction': '4.0'},
        {'ground_truth': '3.0', 'prediction': '6.0'},
    ]
    
    metrics, per_sample = _compute_metrics_property_internal(outputs)
    
    # Should have strong correlation (but not R²=1 due to scale difference)
    assert metrics['pearson'] > 0.99  # Near perfect correlation


def test_mixed_formats():
    """Test extraction from mixed text formats."""
    from src.trainer.metrics.mol_property import _compute_metrics_property_internal
    
    outputs = [
        {'ground_truth': 'The value is 10.5 kcal/mol', 'prediction': '10.2'},
        {'ground_truth': '20.3', 'prediction': 'Predicted: 20.0 eV'},
        {'ground_truth': 'Answer: -5.5', 'prediction': '-5.0'},
    ]
    
    metrics, per_sample = _compute_metrics_property_internal(outputs)
    
    # Should extract numbers correctly from all formats
    assert metrics['valid_ratio'] == 1.0
    assert per_sample[0]['ground_truth_value'] == 10.5
    assert per_sample[0]['prediction_value'] == 10.2
    assert per_sample[2]['ground_truth_value'] == -5.5
    assert per_sample[2]['prediction_value'] == -5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
