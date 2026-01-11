"""Metrics for molecular property prediction tasks."""

import numpy as np
import re


def compute_metrics_mol_property(predictions, labels, tokenizer):
    """
    Compute metrics for molecular property prediction tasks.
    
    Args:
        predictions: List of predicted token ID arrays
        labels: List of ground truth token ID arrays
        tokenizer: Tokenizer for decoding
    
    Returns:
        Dictionary containing:
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - r2: R-squared score
            - pearson: Pearson correlation coefficient
    """
    # Decode predictions and labels
    decoded_preds = []
    decoded_labels = []
    
    for pred, label in zip(predictions, labels):
        # Replace -100 in labels with pad_token_id for decoding
        label_cleaned = np.where(label != -100, label, tokenizer.pad_token_id)
        
        # Decode single sequences
        decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
        decoded_label = tokenizer.decode(label_cleaned, skip_special_tokens=True)
        
        decoded_preds.append(decoded_pred)
        decoded_labels.append(decoded_label)
    
    # Prepare outputs in the format expected by compute_metrics_property_internal
    outputs = []
    for pred, label in zip(decoded_preds, decoded_labels):
        outputs.append({
            'ground_truth': label,
            'prediction': pred,
        })
    
    # Compute metrics using the internal function
    metrics, per_sample = _compute_metrics_property_internal(outputs)
    
    return metrics


def compute_metrics_mol_property_detailed(predictions, labels, tokenizer, prompts=None):
    """
    Compute metrics for molecular property prediction and return detailed per-sample results.
    
    Args:
        predictions: List of predicted token ID arrays
        labels: List of ground truth token ID arrays
        tokenizer: Tokenizer for decoding
        prompts: Optional list of prompt token ID arrays
    
    Returns:
        Tuple of (metrics_dict, detailed_results_list)
    """
    # Decode predictions and labels
    decoded_preds = []
    decoded_labels = []
    decoded_prompts = []
    
    # Decode prompts if provided
    if prompts is not None:
        for prompt_ids in prompts:
            decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            decoded_prompts.append(decoded_prompt)
    
    for pred, label in zip(predictions, labels):
        # Replace -100 in labels with pad_token_id for decoding
        label_cleaned = np.where(label != -100, label, tokenizer.pad_token_id)
        
        # Decode single sequences
        decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
        decoded_label = tokenizer.decode(label_cleaned, skip_special_tokens=True)
        
        decoded_preds.append(decoded_pred)
        decoded_labels.append(decoded_label)
    
    # Prepare outputs in the format expected by compute_metrics_property_internal
    outputs = []
    for pred, label in zip(decoded_preds, decoded_labels):
        outputs.append({
            'ground_truth': label,
            'prediction': pred,
        })
    
    # Compute metrics and get per-sample details
    metrics, per_sample = _compute_metrics_property_internal(outputs)
    
    # Augment per-sample results with prompts if available
    detailed_results = []
    for idx, sample in enumerate(per_sample):
        result = {
            'index': idx,
            **sample
        }
        if decoded_prompts and idx < len(decoded_prompts):
            result['prompt'] = decoded_prompts[idx]
        detailed_results.append(result)
    
    return metrics, detailed_results


def _compute_metrics_property_internal(outputs):
    """
    Internal function to compute metrics for molecular property prediction tasks.
    
    Metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - R2: R-squared score
    - Pearson: Pearson correlation coefficient
    
    Args:
        outputs: List of dicts with 'ground_truth' and 'prediction' keys
    
    Returns:
        Tuple of (metrics_dict, per_sample_list)
    """
    # Handle empty outputs
    if not outputs:
        return {
            'mae': float('inf'),
            'mse': float('inf'),
            'rmse': float('inf'),
            'r2': 0.0,
            'pearson': 0.0,
        }, []
    
    per_sample = []
    gt_values = []
    pred_values = []
    absolute_errors = []
    squared_errors = []
    valid_count = 0
    
    for o in outputs:
        gt_string = o['ground_truth']
        pred_string = o['prediction']
        
        # Extract numerical values from strings
        gt_value = _extract_number(gt_string)
        pred_value = _extract_number(pred_string)
        
        # Check if both values are valid
        valid = (gt_value is not None) and (pred_value is not None)
        
        if valid:
            valid_count += 1
            gt_values.append(gt_value)
            pred_values.append(pred_value)
            
            # Compute per-sample errors
            abs_error = abs(pred_value - gt_value)
            sq_error = (pred_value - gt_value) ** 2
            absolute_errors.append(abs_error)
            squared_errors.append(sq_error)
        else:
            absolute_errors.append(None)
            squared_errors.append(None)
        
        sample_result = {
            'ground_truth': gt_string,
            'prediction': pred_string,
            'ground_truth_value': gt_value,
            'prediction_value': pred_value,
            'absolute_error': absolute_errors[-1],
            'squared_error': squared_errors[-1],
            'valid': valid,
        }
        
        per_sample.append(sample_result)
    
    # Compute aggregated metrics
    if valid_count == 0:
        return {
            'mae': float('inf'),
            'mse': float('inf'),
            'rmse': float('inf'),
            'r2': 0.0,
            'pearson': 0.0,
            'valid_ratio': 0.0,
        }, per_sample
    
    # Convert to numpy arrays for easier computation
    gt_array = np.array(gt_values)
    pred_array = np.array(pred_values)
    
    # MAE (Mean Absolute Error)
    mae = float(np.mean(np.abs(pred_array - gt_array)))
    
    # MSE (Mean Squared Error)
    mse = float(np.mean((pred_array - gt_array) ** 2))
    
    # RMSE (Root Mean Squared Error)
    rmse = float(np.sqrt(mse))
    
    # R2 (R-squared)
    ss_res = np.sum((gt_array - pred_array) ** 2)
    ss_tot = np.sum((gt_array - np.mean(gt_array)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    # Pearson correlation coefficient
    if len(gt_array) > 1:
        correlation_matrix = np.corrcoef(gt_array, pred_array)
        pearson = float(correlation_matrix[0, 1])
        # Handle NaN (can occur if one array is constant)
        if np.isnan(pearson):
            pearson = 0.0
    else:
        pearson = 0.0
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson,
        'valid_ratio': float(valid_count / len(outputs)),
    }
    
    return metrics, per_sample


def _extract_number(text):
    """
    Extract a numerical value from text string.
    
    Handles various formats:
    - Plain numbers: "42", "-3.14", "0.5"
    - With units: "42.5 kcal/mol", "-3.14 eV"
    - In sentences: "The value is 42.5"
    - Scientific notation: "1.5e-3", "2.3E+5"
    
    Args:
        text: String potentially containing a number
    
    Returns:
        Float value if found, None otherwise
    """
    if not text or not isinstance(text, str):
        return None
    
    # Remove common unit suffixes for cleaner extraction
    text_clean = text.strip()
    
    # Pattern to match numbers (int, float, scientific notation, with optional sign)
    # Matches: 42, -3.14, 1.5e-3, -2.3E+5, etc.
    patterns = [
        r'[-+]?\d+\.?\d*[eE][-+]?\d+',  # Scientific notation
        r'[-+]?\d*\.\d+',                # Decimal numbers
        r'[-+]?\d+',                     # Integers
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_clean)
        if matches:
            try:
                # Return the first number found
                return float(matches[0])
            except (ValueError, IndexError):
                continue
    
    return None
