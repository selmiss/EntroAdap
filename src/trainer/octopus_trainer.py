"""
Custom SFT Trainer for Octopus that properly passes graph data.
"""

from typing import Dict, Any, Optional, Union
import torch
from torch.utils.data import DataLoader
from trl import SFTTrainer
import numpy as np


class MultiModalSFTTrainer(SFTTrainer):
    """
    Custom SFT Trainer that overrides compute_loss to pass graph_data
    and related arguments to Octopus.forward().
    
    Also provides custom dataloader with ModalityAwareBatchSampler to ensure
    batches don't mix different modalities (e.g., DNA and RNA).
    """
    
    def __init__(self, *args, compute_text_metrics: bool = False, **kwargs):
        """
        Initialize trainer with optional text metrics computation.
        
        Args:
            compute_text_metrics: If True, compute NLP metrics (BLEU, ROUGE, METEOR) during evaluation
        """
        super().__init__(*args, **kwargs)
        self.compute_text_metrics = compute_text_metrics
        self._eval_predictions = []
        self._eval_labels = []
    def _prepare_dataset(self, dataset, *args, **kwargs):
        """
        Override TRL's dataset preparation to prevent truncation of graph features.
        """

        return dataset

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log to normalize loss for comparison with eval_loss.
        
        The default 'loss' logged by HuggingFace Trainer is the accumulated loss
        over gradient_accumulation_steps, making it incomparable with eval_loss.
        We normalize it by dividing by gradient_accumulation_steps.
        """
        if 'loss' in logs and self.args.gradient_accumulation_steps > 1:
            # Keep original accumulated loss for reference
            logs['loss_accumulated'] = logs['loss']
            # Replace loss with normalized value that's comparable to eval_loss
            logs['loss'] = logs['loss'] / self.args.gradient_accumulation_steps
        
        super().log(logs, start_time=start_time)
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader with a custom batch sampler that groups
        samples by modality to prevent mixing modalities in a single batch.
        """
        from src.data_loader import ModalityAwareBatchSamplerForSFT
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        # Check if dataset has modality information
        has_modality_info = (
            hasattr(train_dataset, 'column_names') and 
            ('modality' in train_dataset.column_names or 'graph_data' in train_dataset.column_names)
        )
        
        if has_modality_info:
            # Use custom batch sampler to group by modality
            batch_sampler = ModalityAwareBatchSamplerForSFT(
                dataset=train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )
            
            # Note: persistent_workers is incompatible with batch_sampler
            return DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                # persistent_workers can't be used with batch_sampler
            )
        else:
            # Fall back to default behavior for datasets without modality
            return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """
        Returns the evaluation DataLoader with a custom batch sampler that groups
        samples by modality to prevent mixing modalities in a single batch.
        """
        from src.data_loader import ModalityAwareBatchSamplerForSFT
        
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        data_collator = self.data_collator
        
        # Check if dataset has modality information
        has_modality_info = (
            hasattr(eval_dataset, 'column_names') and 
            ('modality' in eval_dataset.column_names or 'graph_data' in eval_dataset.column_names)
        )
        
        if has_modality_info:
            # Use custom batch sampler to group by modality
            batch_sampler = ModalityAwareBatchSamplerForSFT(
                dataset=eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,  # Don't shuffle eval data
                seed=self.args.seed,
                drop_last=False,  # Keep all eval samples
            )
            
            # Note: persistent_workers is incompatible with batch_sampler
            return DataLoader(
                eval_dataset,
                batch_sampler=batch_sampler,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                # persistent_workers can't be used with batch_sampler
            )
        else:
            # Fall back to default behavior for datasets without modality
            return super().get_eval_dataloader(eval_dataset)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for multimodal training, passing graph data to model.
        
        Args:
            model: Octopus instance
            inputs: Batch dictionary from MultiModalDataCollator with:
                - input_ids, attention_mask, labels (standard)
                - graph_data, batch, instr_positions, patch_positions (multimodal)
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch
        
        Returns:
            loss (and optionally outputs)
        """
        # Extract multimodal-specific inputs
        graph_data = inputs.pop("graph_data", None)
        batch = inputs.pop("batch", None)
        instr_positions = inputs.pop("instr_positions", None)
        patch_positions = inputs.pop("patch_positions", None)
        patch_mask = inputs.pop("patch_mask", None)
        node_mask = inputs.pop("node_mask", None)
        
        # Remove internal tracking fields
        inputs.pop("_graph_indices", None)
        
        # Call model with all arguments
        outputs = model(
            **inputs,
            graph_data=graph_data,
            batch=batch,
            instr_positions=instr_positions,
            patch_positions=patch_positions,
            patch_mask=patch_mask,
            node_mask=node_mask,
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to collect predictions and labels for text metrics.
        """
        # If text metrics are enabled, we need logits even if prediction_loss_only=True
        need_logits_for_metrics = self.compute_text_metrics and not model.training
        actual_prediction_loss_only = prediction_loss_only and not need_logits_for_metrics
        
        # Call parent prediction_step
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only=actual_prediction_loss_only, ignore_keys=ignore_keys
        )
        
        # If computing text metrics and we're in evaluation mode, collect predictions
        if need_logits_for_metrics and logits is not None:
            # Convert to numpy if needed
            logits_np = logits.cpu().numpy() if isinstance(logits, torch.Tensor) else logits
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            
            # Convert logits to token IDs immediately to avoid storing large logit tensors
            if len(logits_np.shape) == 3:  # (batch_size, seq_len, vocab_size)
                pred_ids = np.argmax(logits_np, axis=-1)
            else:
                pred_ids = logits_np
            
            # Store each sample individually to handle variable lengths
            for pred, label in zip(pred_ids, labels_np):
                self._eval_predictions.append(pred)
                self._eval_labels.append(label)
        
        # Return according to prediction_loss_only flag
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute and log text metrics after evaluation.
        """
        # Reset stored predictions and labels
        self._eval_predictions = []
        self._eval_labels = []
        
        # Run standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # If text metrics are enabled and we collected predictions, compute them
        if self.compute_text_metrics and self._eval_predictions:
            try:
                # Compute text metrics (predictions and labels are lists of variable-length arrays)
                text_metrics = self.compute_metrics_text((self._eval_predictions, self._eval_labels))
                
                # Add metrics to output with appropriate prefix
                for key, value in text_metrics.items():
                    # Skip loss as it's already computed
                    if key != 'loss':
                        output[f"{metric_key_prefix}_{key}"] = value
                
                # Log the text metrics
                self.log(output)
                
                print(f"\n{'='*60}")
                print(f"Text Generation Metrics ({metric_key_prefix}):")
                print(f"{'='*60}")
                for key, value in text_metrics.items():
                    if key != 'loss':
                        print(f"{key:15s}: {value:.4f}")
                print(f"{'='*60}\n")
                
            except Exception as e:
                import traceback
                print(f"Warning: Failed to compute text metrics: {e}")
                traceback.print_exc()
            finally:
                # Clear stored predictions to free memory
                self._eval_predictions = []
                self._eval_labels = []
        
        return output
    
    def compute_metrics_text(self, eval_preds):
        """
        Compute NLP metrics for evaluating generated text.
        
        Args:
            eval_preds: Tuple of (predictions, labels) where:
                - predictions: List of predicted token ID arrays (variable length)
                - labels: List of ground truth token ID arrays (variable length)
        
        Returns:
            Dictionary containing:
                - bleu_2: BLEU-2 score
                - bleu_4: BLEU-4 score
                - rouge_1: ROUGE-1 F1 score
                - rouge_2: ROUGE-2 F1 score
                - rouge_l: ROUGE-L F1 score
                - meteor: METEOR score
        """
        import numpy as np
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
        import nltk
        
        # Download required NLTK data if not already present
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        try:
            nltk.data.find('omw-1.4')
        except LookupError:
            nltk.download('omw-1.4', quiet=True)
        
        predictions, labels = eval_preds
        
        # Get tokenizer from the trainer
        tokenizer = self.tokenizer
        
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
                # Handle edge cases where METEOR computation might fail
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

