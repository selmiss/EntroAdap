"""Metrics for multiple choice QA tasks."""

import re
import numpy as np
from collections import defaultdict


def compute_metrics_multiple_choice(predictions, labels, tokenizer, categories=None):
    """
    Compute accuracy for multiple choice QA with optional per-category breakdown.
    
    Args:
        predictions: List of predicted token ID arrays
        labels: List of ground truth token ID arrays
        tokenizer: Tokenizer for decoding
        categories: Optional list of category/task names for each sample
    
    Returns:
        Dictionary with accuracy metrics (overall and per-category if provided)
    """
    decoded_preds = []
    decoded_labels = []
    
    for pred, label in zip(predictions, labels):
        label_cleaned = np.where(label != -100, label, tokenizer.pad_token_id)
        decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
        decoded_label = tokenizer.decode(label_cleaned, skip_special_tokens=True)
        decoded_preds.append(decoded_pred)
        decoded_labels.append(decoded_label)
    
    pattern = r"[Aa]nswer:\s*([A-Da-d])"
    
    # Track overall and per-category results
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0
    overall_total = 0
    
    for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        # Extract answer from label
        label_match = re.search(pattern, label)
        if not label_match:
            continue
        answer = label_match.group(1).upper()
        
        # Extract prediction from response
        pred_match = re.search(pattern, pred)
        if pred_match:
            prediction = pred_match.group(1).upper()
        else:
            # Fallback: look for A/B/C/D anywhere after "Answer:"
            answer_split = pred.split("Answer:")
            if len(answer_split) > 1:
                answer_part = answer_split[-1].strip()
                if 'A' in answer_part:
                    prediction = 'A'
                elif 'B' in answer_part:
                    prediction = 'B'
                elif 'C' in answer_part:
                    prediction = 'C'
                elif 'D' in answer_part:
                    prediction = 'D'
                else:
                    prediction = 'None'
            else:
                prediction = 'None'
        
        is_correct = (prediction == answer)
        
        # Update overall stats
        if is_correct:
            overall_correct += 1
        overall_total += 1
        
        # Update per-category stats if categories provided
        if categories is not None and idx < len(categories):
            category = categories[idx]
            if is_correct:
                category_stats[category]['correct'] += 1
            category_stats[category]['total'] += 1
    
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    
    # Build output metrics
    metrics = {
        'accuracy': overall_accuracy,
        'correct': overall_correct,
        'total': overall_total,
    }
    
    # Add per-category metrics if available
    if category_stats:
        for category, stats in sorted(category_stats.items()):
            cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            metrics[f'accuracy_{category}'] = cat_accuracy
            metrics[f'correct_{category}'] = stats['correct']
            metrics[f'total_{category}'] = stats['total']
    
    return metrics


def compute_metrics_multiple_choice_detailed(predictions, labels, tokenizer, categories=None, prompts=None):
    """
    Compute accuracy for multiple choice QA and return detailed per-sample results.
    
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
    
    # Decode prompts if provided
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
    
    pattern = r"[Aa]nswer:\s*([A-Da-d])"
    
    # Track overall and per-category results
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0
    overall_total = 0
    detailed_results = []
    
    for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        # Extract answer from label
        label_match = re.search(pattern, label)
        if not label_match:
            continue
        answer = label_match.group(1).upper()
        
        # Extract prediction from response
        pred_match = re.search(pattern, pred)
        if pred_match:
            prediction = pred_match.group(1).upper()
        else:
            # Fallback: look for A/B/C/D anywhere after "Answer:"
            answer_split = pred.split("Answer:")
            if len(answer_split) > 1:
                answer_part = answer_split[-1].strip()
                if 'A' in answer_part:
                    prediction = 'A'
                elif 'B' in answer_part:
                    prediction = 'B'
                elif 'C' in answer_part:
                    prediction = 'C'
                elif 'D' in answer_part:
                    prediction = 'D'
                else:
                    prediction = 'None'
            else:
                prediction = 'None'
        
        is_correct = (prediction == answer)
        
        # Build detailed result
        result = {
            'index': idx,
            'prediction': pred,
            'extracted_answer': prediction,
            'ground_truth_answer': answer,
            'correct': is_correct,
        }
        if categories is not None and idx < len(categories):
            result['category'] = categories[idx]
        if decoded_prompts and idx < len(decoded_prompts):
            result['prompt'] = decoded_prompts[idx]
        
        detailed_results.append(result)
        
        # Update overall stats
        if is_correct:
            overall_correct += 1
        overall_total += 1
        
        # Update per-category stats if categories provided
        if categories is not None and idx < len(categories):
            category = categories[idx]
            if is_correct:
                category_stats[category]['correct'] += 1
            category_stats[category]['total'] += 1
    
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    
    # Build output metrics
    metrics = {
        'accuracy': overall_accuracy,
        'correct': overall_correct,
        'total': overall_total,
    }
    
    # Add per-category metrics if available
    if category_stats:
        for category, stats in sorted(category_stats.items()):
            cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            metrics[f'accuracy_{category}'] = cat_accuracy
            metrics[f'correct_{category}'] = stats['correct']
            metrics[f'total_{category}'] = stats['total']
    
    return metrics, detailed_results

