"""Metrics for molecular generation tasks."""

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
import selfies

def compute_metrics_molecule_generation(predictions, labels, tokenizer, selfies_mode=True):
    """
    Compute metrics for molecule generation tasks.
    
    Args:
        predictions: List of predicted token ID arrays
        labels: List of ground truth token ID arrays
        tokenizer: Tokenizer for decoding
        selfies_mode: Whether the inputs are SELFIES strings (True) or SMILES (False)
    
    Returns:
        Dictionary containing:
            - exact: Exact match accuracy
            - bleu: BLEU score (on string tokens)
            - levenshtein: Average Levenshtein distance
            - rdk_fts: RDKit fingerprint Tanimoto similarity
            - maccs_fts: MACCS fingerprint Tanimoto similarity
            - morgan_fts: Morgan fingerprint Tanimoto similarity
            - validity: Valid SMILES percentage
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
    
    # Prepare outputs in the format expected by compute_metrics_reaction
    outputs = []
    for pred, label in zip(decoded_preds, decoded_labels):
        outputs.append({
            'ground_truth': label,
            'prediction': pred,
        })
    
    # Compute metrics using the reaction metrics function
    metrics, per_sample = _compute_metrics_reaction_internal(outputs, selfies_mode)
    
    return metrics


def compute_metrics_molecule_generation_detailed(predictions, labels, tokenizer, selfies_mode=True, prompts=None):
    """
    Compute metrics for molecule generation and return detailed per-sample results.
    
    Args:
        predictions: List of predicted token ID arrays
        labels: List of ground truth token ID arrays
        tokenizer: Tokenizer for decoding
        selfies_mode: Whether the inputs are SELFIES strings (True) or SMILES (False)
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
    
    # Prepare outputs in the format expected by compute_metrics_reaction
    outputs = []
    for pred, label in zip(decoded_preds, decoded_labels):
        outputs.append({
            'ground_truth': label,
            'prediction': pred,
        })
    
    # Compute metrics and get per-sample details
    metrics, per_sample = _compute_metrics_reaction_internal(outputs, selfies_mode)
    
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


def _compute_metrics_reaction_internal(outputs, selfies_mode=True):
    """
    Internal function to compute metrics for reaction/molecule generation tasks.
    
    Metrics:
    - Exact: Exact match accuracy
    - BLEU: BLEU score
    - Levenshtein: Levenshtein distance
    - RDK FTS: RDKit fingerprint Tanimoto similarity
    - MACC FTS: MACCS fingerprint Tanimoto similarity
    - Morgan FTS: Morgan fingerprint Tanimoto similarity
    - Validity: Valid SMILES percentage
    
    Args:
        outputs: List of dicts with 'ground_truth' and 'prediction' keys
        selfies_mode: Whether the inputs are SELFIES strings (True) or SMILES (False)
    
    Returns:
        Tuple of (metrics_dict, per_sample_list)
    """
    # Handle empty outputs
    if not outputs:
        return {
            'exact': 0.0,
            'bleu': 0.0,
            'levenshtein': float('inf'),
            'rdk_fts': 0.0,
            'maccs_fts': 0.0,
            'morgan_fts': 0.0,
            'validity': 0.0,
        }, []
    
    per_sample = []
    exact_matches = []
    bleu_scores = []
    lev_distances = []
    rdk_similarities = []
    maccs_similarities = []
    morgan_similarities = []
    valid_count = 0
    total_count = len(outputs)
    
    for o in outputs:
        gt_string = o['ground_truth']
        pred_string = o['prediction']
        
        # Convert SELFIES to SMILES if needed
        if selfies_mode:
            gt_smiles = _selfies_to_smiles(gt_string)
            pred_smiles = _selfies_to_smiles(pred_string)
        else:
            # In SMILES mode, validate the SMILES strings
            gt_smiles = _validate_smiles(gt_string)
            pred_smiles = _validate_smiles(pred_string)
        
        # Exact match (on original strings)
        exact_match = int(gt_string == pred_string)
        exact_matches.append(exact_match)
        
        # BLEU score (on strings as character tokens)
        try:
            ref_tokens = list(gt_string)
            pred_tokens = list(pred_string)
            bleu = corpus_bleu([[ref_tokens]], [pred_tokens], weights=(0.5, 0.5))
            bleu_scores.append(bleu)
        except Exception:
            bleu_scores.append(0.0)
        
        # Levenshtein distance
        try:
            lev_dist = _levenshtein_distance(gt_string, pred_string)
            lev_distances.append(lev_dist)
        except Exception:
            lev_distances.append(len(gt_string))  # Worst case
        
        # Fingerprint similarities (only if both SMILES are valid)
        pred_valid = (pred_smiles is not None)
        gt_valid = (gt_smiles is not None)
        
        if pred_valid:
            valid_count += 1
        
        if gt_valid and pred_valid:
            rdk_sim = _compute_fingerprint_similarity(gt_smiles, pred_smiles, 'rdk')
            maccs_sim = _compute_fingerprint_similarity(gt_smiles, pred_smiles, 'maccs')
            morgan_sim = _compute_fingerprint_similarity(gt_smiles, pred_smiles, 'morgan')
            rdk_similarities.append(rdk_sim)
            maccs_similarities.append(maccs_sim)
            morgan_similarities.append(morgan_sim)
        else:
            # Invalid SMILES - assign 0 similarity
            rdk_similarities.append(0.0)
            maccs_similarities.append(0.0)
            morgan_similarities.append(0.0)
        
        sample_result = {
            'ground_truth': gt_string,
            'prediction': pred_string,
            'ground_truth_smiles': gt_smiles,
            'prediction_smiles': pred_smiles,
            'exact_match': exact_match,
            'bleu': bleu_scores[-1],
            'levenshtein': lev_distances[-1],
            'rdk_fts': rdk_similarities[-1],
            'maccs_fts': maccs_similarities[-1],
            'morgan_fts': morgan_similarities[-1],
        }
        
        if selfies_mode:
            sample_result['ground_truth_selfies'] = gt_string
            sample_result['prediction_selfies'] = pred_string
        
        per_sample.append(sample_result)
    
    # Compute aggregated metrics
    metrics = {
        'exact': float(np.mean(exact_matches)),
        'bleu': float(np.mean(bleu_scores)),
        'levenshtein': float(np.mean(lev_distances)),
        'rdk_fts': float(np.mean(rdk_similarities)),
        'maccs_fts': float(np.mean(maccs_similarities)),
        'morgan_fts': float(np.mean(morgan_similarities)),
        'validity': float(valid_count / total_count) if total_count > 0 else 0.0,
    }
    
    return metrics, per_sample


def _selfies_to_smiles(selfies_str):
    """Convert SELFIES string to SMILES string."""
    try:
        
        smiles = selfies.decoder(selfies_str)
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return smiles
    except Exception:
        return None


def _validate_smiles(smiles_str):
    """Validate a SMILES string and return it if valid, None otherwise."""
    try:
        if not smiles_str:
            return None
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None
        return smiles_str
    except Exception:
        return None


def _levenshtein_distance(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer than s2
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def _compute_fingerprint_similarity(smiles1, smiles2, fp_type='morgan'):
    """
    Compute Tanimoto similarity between two SMILES using specified fingerprint.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        fp_type: Fingerprint type - 'morgan', 'rdk', or 'maccs'
    
    Returns:
        Tanimoto similarity score (float between 0 and 1)
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        if fp_type == 'morgan':
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        elif fp_type == 'rdk':
            fp1 = Chem.RDKFingerprint(mol1)
            fp2 = Chem.RDKFingerprint(mol2)
        elif fp_type == 'maccs':
            fp1 = MACCSkeys.GenMACCSKeys(mol1)
            fp2 = MACCSkeys.GenMACCSKeys(mol2)
        else:
            return 0.0
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0
