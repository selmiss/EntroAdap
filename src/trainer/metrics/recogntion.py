"""Metrics for entity recognition tasks."""

import numpy as np
from collections import defaultdict


def compute_metrics_recognition(predictions, labels, tokenizer, categories=None):
    """
    Compute F1 for entity recognition tasks.
    
    Args:
        predictions: List of predicted token ID arrays
        labels: List of ground truth token ID arrays
        tokenizer: Tokenizer for decoding
        categories: Optional list of category/task names for each sample
    
    Returns:
        Dictionary with F1 metrics (overall and per-category if provided)
    """
    decoded_preds = []
    decoded_labels = []
    
    for pred, label in zip(predictions, labels):
        label_cleaned = np.where(label != -100, label, tokenizer.pad_token_id)
        decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
        decoded_label = tokenizer.decode(label_cleaned, skip_special_tokens=True)
        decoded_preds.append(decoded_pred)
        decoded_labels.append(decoded_label)
    
    category_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        label_entities = set(e.strip().lower() for e in label.split(',') if e.strip())
        pred_lower = pred.lower()
        
        tp = sum(1 for entity in label_entities if entity in pred_lower)
        fn = len(label_entities) - tp
        fp = 0 if tp > 0 else 1
        
        overall_tp += tp
        overall_fn += fn
        overall_fp += fp
        
        if categories is not None and idx < len(categories):
            category = categories[idx]
            category_stats[category]['tp'] += tp
            category_stats[category]['fn'] += fn
            category_stats[category]['fp'] += fp
    
    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': overall_tp,
        'fp': overall_fp,
        'fn': overall_fn,
    }
    
    if category_stats:
        for category, stats in sorted(category_stats.items()):
            cat_precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0.0
            cat_recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0.0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0.0
            metrics[f'f1_{category}'] = cat_f1
            metrics[f'precision_{category}'] = cat_precision
            metrics[f'recall_{category}'] = cat_recall
    
    return metrics


def compute_metrics_recognition_detailed(predictions, labels, tokenizer, categories=None, prompts=None):
    """
    Compute F1 for entity recognition and return detailed per-sample results.
    
    Args:
        predictions: List of predicted token ID arrays
        labels: List of ground truth token ID arrays
        tokenizer: Tokenizer for decoding
        categories: Optional list of category/task names for each sample
        prompts: Optional list of prompt token ID arrays
    
    Returns:
        Tuple of (metrics_dict, detailed_results_list)
    """
    decoded_preds = []
    decoded_labels = []
    decoded_prompts = []
    
    if prompts is not None:
        for prompt_ids in prompts:
            decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            decoded_prompts.append(decoded_prompt)
    
    for pred, label in zip(predictions, labels):
        label_cleaned = np.where(label != -100, label, tokenizer.pad_token_id)
        decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
        decoded_label = tokenizer.decode(label_cleaned, skip_special_tokens=True)
        decoded_preds.append(decoded_pred)
        decoded_labels.append(decoded_label)
    
    category_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    detailed_results = []
    
    for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        label_entities = set(e.strip().lower() for e in label.split(',') if e.strip())
        pred_lower = pred.lower()
        
        matched_entities = [entity for entity in label_entities if entity in pred_lower]
        tp = len(matched_entities)
        fn = len(label_entities) - tp
        fp = 0 if tp > 0 else 1
        
        result = {
            'index': idx,
            'prediction': pred,
            'ground_truth': label,
            'matched_entities': matched_entities,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }
        if categories is not None and idx < len(categories):
            result['category'] = categories[idx]
        if decoded_prompts and idx < len(decoded_prompts):
            result['prompt'] = decoded_prompts[idx]
        
        detailed_results.append(result)
        
        overall_tp += tp
        overall_fn += fn
        overall_fp += fp
        
        if categories is not None and idx < len(categories):
            category = categories[idx]
            category_stats[category]['tp'] += tp
            category_stats[category]['fn'] += fn
            category_stats[category]['fp'] += fp
    
    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': overall_tp,
        'fp': overall_fp,
        'fn': overall_fn,
    }
    
    if category_stats:
        for category, stats in sorted(category_stats.items()):
            cat_precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0.0
            cat_recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0.0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0.0
            metrics[f'f1_{category}'] = cat_f1
            metrics[f'precision_{category}'] = cat_precision
            metrics[f'recall_{category}'] = cat_recall
    
    return metrics, detailed_results
