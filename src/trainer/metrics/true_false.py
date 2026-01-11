"""Metrics for true/false (yes/no) questions."""

import re
import numpy as np
from collections import defaultdict


def compute_metrics_true_false(predictions, labels, tokenizer, categories=None):
    """
    Compute accuracy for true/false (yes/no) questions with optional per-category breakdown.
    
    The evaluator extracts answers by:
    1. First checking if the first token is "Yes"/"No"/"True"/"False"
    2. If not, searching for the first occurrence of "yes"/"no"/"true"/"false" in the text
    
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
    
    # Track overall and per-category results
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0
    overall_total = 0
    
    for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        # Extract answer from label
        answer = extract_yes_no_answer(label)
        if answer is None:
            continue
        
        # Extract prediction from response
        prediction = extract_yes_no_answer(pred)
        if prediction is None:
            prediction = 'unknown'
        
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


def compute_metrics_true_false_detailed(predictions, labels, tokenizer, categories=None, prompts=None):
    """
    Compute accuracy for true/false questions and return detailed per-sample results.
    
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
    
    # Track overall and per-category results
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0
    overall_total = 0
    detailed_results = []
    
    for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        # Extract answer from label
        answer = extract_yes_no_answer(label)
        if answer is None:
            continue
        
        # Extract prediction from response
        prediction = extract_yes_no_answer(pred)
        if prediction is None:
            prediction = 'unknown'
        
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


def extract_yes_no_answer(text):
    """
    Extract yes/no or true/false answer from text.
    
    Strategy:
    1. Check if the first token (word) is Yes/No/True/False
    2. If not, find the first occurrence of yes/no/true/false in the text
    
    Args:
        text: String to extract answer from
    
    Returns:
        'yes', 'no', or None if no answer found
    """
    if not text:
        return None
    
    text_stripped = text.strip()
    if not text_stripped:
        return None
    
    # Extract first token (split by whitespace and punctuation)
    first_token_match = re.match(r'^([a-zA-Z]+)', text_stripped)
    if first_token_match:
        first_token = first_token_match.group(1).lower()
        if first_token in ['yes', 'true']:
            return 'yes'
        elif first_token in ['no', 'false']:
            return 'no'
    
    # If first token check failed, search for first occurrence of yes/no/true/false
    # Use case-insensitive word boundary matching
    yes_match = re.search(r'\b(yes|true)\b', text, re.IGNORECASE)
    no_match = re.search(r'\b(no|false)\b', text, re.IGNORECASE)
    
    # Return the one that appears first
    if yes_match and no_match:
        if yes_match.start() < no_match.start():
            return 'yes'
        else:
            return 'no'
    elif yes_match:
        return 'yes'
    elif no_match:
        return 'no'
    
    return None
