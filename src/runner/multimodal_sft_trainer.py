"""
Custom SFT Trainer for MultiModalLLM that properly passes graph data.
"""

from typing import Dict, Any, Optional
import torch
from trl import SFTTrainer


class MultiModalSFTTrainer(SFTTrainer):
    """
    Custom SFT Trainer that overrides compute_loss to pass graph_data
    and related arguments to MultiModalLLM.forward().
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for multimodal training, passing graph data to model.
        
        Args:
            model: MultiModalLLM instance
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

