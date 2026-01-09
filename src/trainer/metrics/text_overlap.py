"""Metrics for text generation evaluation using overlap-based measures."""

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk


def compute_metrics_text_overlap(predictions, labels, tokenizer):
    """
    Compute NLP metrics for evaluating generated text.
    
    Args:
        predictions: List of predicted token ID arrays (variable length)
        labels: List of ground truth token ID arrays (variable length)
        tokenizer: Tokenizer for decoding token IDs to text
    
    Returns:
        Dictionary containing:
            - bleu_2: BLEU-2 score
            - bleu_4: BLEU-4 score
            - rouge_1: ROUGE-1 F1 score
            - rouge_2: ROUGE-2 F1 score
            - rouge_l: ROUGE-L F1 score
            - meteor: METEOR score
    """
    # Download required NLTK data if not already present
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    
    # Decode predictions and labels (already token IDs, just need to decode)
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
    
    # Initialize metrics
    bleu_2_scores = []
    bleu_4_scores = []
    meteor_scores = []
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    # Smoothing function for BLEU to handle edge cases
    smooth_fn = SmoothingFunction().method1
    
    # Compute metrics for each sample
    for pred, label in zip(decoded_preds, decoded_labels):
        # Tokenize for BLEU and METEOR
        pred_tokens = pred.split()
        label_tokens = label.split()
        
        # Skip empty predictions or labels
        if not pred_tokens or not label_tokens:
            continue
        
        # BLEU scores (need reference as list of lists)
        reference = [label_tokens]
        bleu_2 = sentence_bleu(reference, pred_tokens, weights=(0.5, 0.5), 
                               smoothing_function=smooth_fn)
        bleu_4 = sentence_bleu(reference, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                               smoothing_function=smooth_fn)
        bleu_2_scores.append(bleu_2)
        bleu_4_scores.append(bleu_4)
        
        # METEOR score
        try:
            meteor = meteor_score([label_tokens], pred_tokens)
            meteor_scores.append(meteor)
        except Exception:
            pass
        
        # ROUGE scores
        rouge_scores = scorer.score(label, pred)
        rouge_1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge_2_scores.append(rouge_scores['rouge2'].fmeasure)
        rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
    
    # Return averaged metrics
    metrics = {
        'bleu_2': float(np.mean(bleu_2_scores)) if bleu_2_scores else 0.0,
        'bleu_4': float(np.mean(bleu_4_scores)) if bleu_4_scores else 0.0,
        'rouge_1': float(np.mean(rouge_1_scores)) if rouge_1_scores else 0.0,
        'rouge_2': float(np.mean(rouge_2_scores)) if rouge_2_scores else 0.0,
        'rouge_l': float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
        'meteor': float(np.mean(meteor_scores)) if meteor_scores else 0.0,
    }
    
    return metrics

