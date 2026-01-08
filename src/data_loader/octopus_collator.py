# Multi-modal data collator for training with text and graph structures

import logging
import torch
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy
from datasets import Dataset, DatasetDict

from .graph_batch_utils import merge_protein_graphs, merge_molecule_graphs


logger = logging.getLogger(__name__)


@dataclass
class MultiModalDataCollator:
    """
    Collates batches with text and graph structures for Octopus training.
    
    Works with SFTTrainer by handling pre-tokenized inputs and batching graph data.
    Produces outputs compatible with Octopus.forward() signature:
    - input_ids, attention_mask, labels: [B, seq_len]
    - graph_data: {'modality': str, 'value': {node_feat, edge_index, pos, ...}}
    - batch: [N] node-to-graph assignment
    - instr_positions: [B, max_instr_len] token positions for full prompt (system + user messages)
    - patch_positions: [B, 1] single position per sample where patches should be injected
    """
    
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    max_instr_positions: int = 32  # Max instruction positions to keep per sample
    structure_tokens: List[str] = None  # List of structure tokens to try (e.g., ["<STRUCTURE>", "<mol>"])
    insert_structure_if_missing: bool = True  # Insert structure token at end of user query if not found
    
    def __post_init__(self):
        """Set default structure tokens if not provided."""
        if self.structure_tokens is None:
            self.structure_tokens = ["<STRUCTURE>", "<mol>", "<DNA>", "<RNA>"]
    
    @staticmethod
    def _as_tensor(x: Any, *, dtype: torch.dtype) -> torch.Tensor:
        """Convert nested python / numpy structures to torch tensors (idempotent for tensors)."""
        if x is None:
            raise ValueError("Graph field is None; cannot convert to tensor.")
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        return torch.tensor(x, dtype=dtype)
    
    @staticmethod
    def _normalize_edge_index(ei: torch.Tensor) -> torch.Tensor:
        """
        Ensure edge_index shape is [2, E].
        Accepts [2, E] or [E, 2] (will transpose).
        """
        if ei.dim() != 2:
            raise ValueError(f"edge_index must be 2D, got shape={tuple(ei.shape)}")
        if ei.size(0) == 2:
            return ei
        if ei.size(1) == 2:
            return ei.t().contiguous()
        raise ValueError(f"edge_index must be [2, E] or [E, 2], got shape={tuple(ei.shape)}")
    
    def _pad_text_features(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pad text sequences and return batch."""
        # Extract text features (without labels - tokenizer doesn't handle labels)
        text_features = []
        has_labels = "labels" in features[0]
        
        for f in features:
            text_feat = {
                "input_ids": f["input_ids"],
                "attention_mask": f.get("attention_mask", [1] * len(f["input_ids"])),
            }
            text_features.append(text_feat)
        
        # Use tokenizer to pad input_ids and attention_mask
        # Use 'longest' padding strategy to avoid max_length warnings
        padding_strategy = 'longest' if self.padding is True else self.padding
        batch = self.tokenizer.pad(
            text_features,
            padding=padding_strategy,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Manually pad labels if present (tokenizer can't handle this)
        if has_labels:
            labels = [f["labels"] for f in features]
            max_len = batch["input_ids"].shape[1]
            padded_labels = []
            for label in labels:
                pad_len = max_len - len(label)
                padded_labels.append(label + [-100] * pad_len)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch
    
    def _collate_graph_data(self, features: List[Dict[str, Any]]) -> Optional[tuple]:
        """
        Collate graph structures from multiple examples.
        
        Supports two formats:
        1. Pre-built graph_data dicts (from MultiModalSFTDataset)
        2. Raw graph columns (from HF datasets: modality, node_feat, pos, edge_index, etc.)
        
        Returns:
            (graph_data, batch_tensor, graph_indices) or None if no graphs present
            - graph_data: {'modality': str, 'value': {...}} batched graph
            - batch_tensor: [N] node-to-graph assignment
            - graph_indices: List[int] indices of which examples have graphs
        """
        # Check if we have raw graph columns (HF dataset format)
        has_raw_columns = (
            len(features) > 0 and 
            'modality' in features[0] and 
            'node_feat' in features[0] and 
            'pos' in features[0]
        )
        
        if has_raw_columns:
            # Fast path: Work directly with raw columns (no intermediate graph_data)
            graph_values = []
            graph_indices = []
            modalities = []
            
            for i, f in enumerate(features):
                # Check if this feature has valid graph data
                if f.get('node_feat') is not None and f.get('pos') is not None:
                    modalities.append(f['modality'])
                    graph_indices.append(i)
                    
                    # Build value dict directly from raw columns
                    value: Dict[str, torch.Tensor] = {
                        "node_feat": self._as_tensor(f["node_feat"], dtype=torch.float32),
                        "pos": self._as_tensor(f["pos"], dtype=torch.float32),
                    }
                    
                    # Add edges if present
                    if "edge_index" in f and f["edge_index"] is not None:
                        ei = self._as_tensor(f["edge_index"], dtype=torch.long)
                        if ei.numel() > 0:
                            value["edge_index"] = self._normalize_edge_index(ei)
                    
                    if "edge_attr" in f and f["edge_attr"] is not None:
                        value["edge_attr"] = self._as_tensor(f["edge_attr"], dtype=torch.float32)
                    
                    if "edge_feat_dist" in f and f["edge_feat_dist"] is not None:
                        value["edge_feat_dist"] = self._as_tensor(f["edge_feat_dist"], dtype=torch.float32)
                    
                    if "chem_edge_index" in f and f["chem_edge_index"] is not None:
                        cei = self._as_tensor(f["chem_edge_index"], dtype=torch.long)
                        if cei.numel() > 0:
                            value["chem_edge_index"] = self._normalize_edge_index(cei)
                    
                    if "chem_edge_feat_cat" in f and f["chem_edge_feat_cat"] is not None:
                        value["chem_edge_feat_cat"] = self._as_tensor(f["chem_edge_feat_cat"], dtype=torch.long)
                    
                    graph_values.append(value)
            
            if not graph_values:
                return None
            
            # Check modality consistency
            if len(set(modalities)) > 1:
                modality_counts = {}
                for m in modalities:
                    modality_counts[m] = modality_counts.get(m, 0) + 1
                raise ValueError(
                    f"Mixed modalities in batch not supported. Found: {modality_counts}. "
                    f"Ensure batches contain only one modality (use ModalityAwareBatchSampler)."
                )
            
            modality = modalities[0]
            
        else:
            # Slow path: Pre-built graph_data dicts (backward compatibility)
            graphs = []
            graph_indices = []
            for i, f in enumerate(features):
                if 'graph_data' in f and f['graph_data'] is not None:
                    graphs.append(f['graph_data'])
                    graph_indices.append(i)
            
            if not graphs:
                return None
            
            # Group by modality
            modalities = [g['modality'] for g in graphs]
            if len(set(modalities)) > 1:
                modality_counts = {}
                for m in modalities:
                    modality_counts[m] = modality_counts.get(m, 0) + 1
                raise ValueError(
                    f"Mixed modalities in batch not supported. Found: {modality_counts}. "
                    f"Ensure dataset is sorted by modality before batching, or use a custom sampler."
                )
            
            modality = modalities[0]
            graph_values = [g['value'] for g in graphs]
            
            # Convert to tensors
            normalized_values: List[Dict[str, torch.Tensor]] = []
            for gv in graph_values:
                if not isinstance(gv, dict):
                    raise ValueError(f"graph_data['value'] must be a dict, got {type(gv)}")
                
                out: Dict[str, torch.Tensor] = {}
                out["node_feat"] = self._as_tensor(gv["node_feat"], dtype=torch.float32)
                out["pos"] = self._as_tensor(gv["pos"], dtype=torch.float32)
                
                if "edge_index" in gv and gv["edge_index"] is not None:
                    ei = self._as_tensor(gv["edge_index"], dtype=torch.long)
                    out["edge_index"] = self._normalize_edge_index(ei)
                
                if "edge_attr" in gv and gv["edge_attr"] is not None:
                    out["edge_attr"] = self._as_tensor(gv["edge_attr"], dtype=torch.float32)
                
                if "edge_feat_dist" in gv and gv["edge_feat_dist"] is not None:
                    out["edge_feat_dist"] = self._as_tensor(gv["edge_feat_dist"], dtype=torch.float32)
                
                if "chem_edge_index" in gv and gv["chem_edge_index"] is not None:
                    cei = self._as_tensor(gv["chem_edge_index"], dtype=torch.long)
                    out["chem_edge_index"] = self._normalize_edge_index(cei)
                
                if "chem_edge_feat_cat" in gv and gv["chem_edge_feat_cat"] is not None:
                    out["chem_edge_feat_cat"] = self._as_tensor(gv["chem_edge_feat_cat"], dtype=torch.long)
                
                normalized_values.append(out)
            
            graph_values = normalized_values
        
        # Merge graphs using modality-specific logic
        if modality == 'protein':
            if any("edge_index" not in gv for gv in graph_values):
                raise ValueError("Protein modality requires `edge_index` in graph_data['value'].")
            merged = merge_protein_graphs(graph_values)
        else:  # molecule, DNA, RNA
            merged = merge_molecule_graphs(graph_values)
        
        batched_graph = {
            'modality': modality,
            'value': merged,
        }
        
        batch_tensor = merged['batch']
        
        return batched_graph, batch_tensor, graph_indices
    
    def _add_instr_positions(self, batch: Dict[str, Any], features: List[Dict[str, Any]]) -> None:
        """Add padded instruction positions to batch.

        Supports either:
        - instr_positions: explicit list of token indices
        - instr_len: integer prefix length (positions are generated as [0..instr_len-1])
        """
        if not any(('instr_positions' in f) or ('instr_len' in f) for f in features):
            return
        
        # Get instruction positions from each feature
        instr_pos_list = []
        for f in features:
            if 'instr_positions' in f:
                pos = f.get('instr_positions', [-1])
            else:
                instr_len = int(f.get('instr_len', 0) or 0)
                pos = list(range(min(instr_len, self.max_instr_positions))) if instr_len > 0 else [-1]
            # Truncate to max length
            pos = pos[:self.max_instr_positions]
            instr_pos_list.append(pos)
        
        # Find max length
        max_len = max(len(p) for p in instr_pos_list)
        
        # Pad with -1
        padded_positions = []
        for pos in instr_pos_list:
            padded = pos + [-1] * (max_len - len(pos))
            padded_positions.append(padded)
        
        batch['instr_positions'] = torch.tensor(padded_positions, dtype=torch.long)
    
    def _add_patch_positions(self, batch: Dict[str, Any], features: List[Dict[str, Any]]) -> None:
        """Add patch positions to batch (single position per sample where graph patches should be injected).
        
        patch_position indicates the index where <STRUCTURE> token appears.
        The model will INSERT k_max patches at this position (not replace).
        """
        if not any('patch_position' in f for f in features):
            return
        
        # Get patch position from each feature (-1 if no graph data)
        patch_pos_list = []
        for f in features:
            pos = f.get('patch_position', -1)
            patch_pos_list.append(pos)
        
        batch['patch_positions'] = torch.tensor(patch_pos_list, dtype=torch.long).unsqueeze(-1)  # [B, 1]
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of examples with text and graph structures.
        
        Returns batch compatible with Octopus.forward():
        - input_ids: [B, seq_len]
        - attention_mask: [B, seq_len]
        - labels: [B, seq_len]
        - graph_data: {'modality': str, 'value': {...}} (if graphs present)
        - batch: [N] node-to-graph assignment (if graphs present)
        - instr_positions: [B, max_instr_len] (if present)
        - patch_positions: [B, 1] single position where patches should be inserted (if present)
        """
        # Pad text features (input_ids, labels, attention_mask)
        batch = self._pad_text_features(features)
        
        # Collate graph data if present
        graph_result = self._collate_graph_data(features)
        if graph_result is not None:
            graph_data, batch_tensor, graph_indices = graph_result
            batch['graph_data'] = graph_data
            batch['batch'] = batch_tensor
            
            # Store which examples in batch have graphs
            # (useful if supporting mixed batches in future)
            batch['_graph_indices'] = graph_indices
        
        # Add instruction positions
        self._add_instr_positions(batch, features)
        
        # Add patch positions
        self._add_patch_positions(batch, features)
        
        return batch


def preprocess_multimodal_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    max_seq_length: int = 2048,
    structure_tokens: List[str] = None,
    insert_structure_if_missing: bool = True,
    max_atoms: Optional[int] = None,
    max_edges: Optional[int] = None,
    filter_before_tokenization: bool = False,
) -> DatasetDict:
    """
    Preprocess multimodal dataset by tokenizing messages and creating labels.
    
    This function:
    1. (Optional) Filters out samples that exceed max_atoms or max_edges thresholds
    2. Applies chat template to convert messages to text
    3. Tokenizes the text into input_ids
    4. Creates labels (copies of input_ids for causal LM training)
    5. Finds structure token positions for patch injection (supports custom tokens)
    6. If no structure token found and insert_structure_if_missing=True, inserts at end of user query
    7. Computes instruction positions (full prompt: system + user messages before first assistant response)
    8. Preserves graph_data and other fields from dataset
    
    Args:
        dataset: Dataset dictionary containing the data
        tokenizer: Tokenizer to use for processing (must have structure tokens registered)
        split: Split name to process (e.g., "train")
        max_seq_length: Maximum sequence length for truncation
        structure_tokens: List of structure tokens to search for (e.g., ["<STRUCTURE>", "<mol>"])
        insert_structure_if_missing: If True, insert structure token at end of user query when not found
        max_atoms: Optional maximum number of atoms per structure. Only used if filter_before_tokenization=True.
        max_edges: Optional maximum number of edges per structure. Only used if filter_before_tokenization=True.
        filter_before_tokenization: If True, filter dataset before tokenization (can be slow).
            If False (default), filtering happens during training via skip_on_error in collator.
    
    Returns:
        Processed dataset with tokenized inputs, labels, patch_position, and instr_len
    """
    if structure_tokens is None:
        structure_tokens = ["<STRUCTURE>"]
    def _preprocess_batch(examples):
        """Tokenize messages and prepare labels while preserving multimodal fields."""
        # Check if examples have 'messages' or if they're already tokenized
        if 'messages' not in examples and 'input_ids' in examples:
            # Already preprocessed, just return
            return examples
        
        # Apply chat template to messages
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        
        # Tokenize full texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # Padding handled by collator
        )
        
        # For causal LM, labels = input_ids (copy each list in the batch)
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        
        # Find <STRUCTURE> token position for patch injection
        # Support multiple structure tokens (try each in order)
        patch_position_batch: List[int] = []
        for idx, input_ids in enumerate(tokenized["input_ids"]):
            # Try each structure token in the list
            patch_pos = -1
            found_token = None
            for structure_token in structure_tokens:
                structure_token_id = tokenizer.convert_tokens_to_ids(structure_token)
                if structure_token_id is not None and structure_token_id in input_ids:
                    patch_pos = input_ids.index(structure_token_id)
                    found_token = structure_token
                    break
            
            # Fallback: Insert structure token at end of user query if not found
            # BUT ONLY if this sample actually has graph data!
            if patch_pos == -1 and insert_structure_if_missing:
                # Check if this sample has graph data
                has_graph = False
                if 'node_feat' in examples and idx < len(examples['node_feat']):
                    # Raw column format
                    has_graph = examples['node_feat'][idx] is not None
                elif 'graph_data' in examples and idx < len(examples['graph_data']):
                    # Nested graph_data format
                    has_graph = examples['graph_data'][idx] is not None
                
                # Only insert structure token if graph data exists
                if has_graph:
                    # Find end of last user message (before first assistant response)
                    messages = examples["messages"][idx]
                    first_assistant_idx = None
                    last_user_idx = None
                    for i, msg in enumerate(messages):
                        if isinstance(msg, dict):
                            if msg.get("role") == "user":
                                last_user_idx = i
                            elif msg.get("role") == "assistant":
                                first_assistant_idx = i
                                break
                    
                    if last_user_idx is not None:
                        # Reconstruct text up to and including last user message
                        user_messages = messages[:last_user_idx + 1]
                        user_text = tokenizer.apply_chat_template(
                            user_messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                        
                        # Insert structure token before closing the user message
                        # For ChatML format: insert before </im_end>
                        if "<|im_end|>" in user_text:
                            # Insert before the last </im_end> (which closes the user message)
                            parts = user_text.rsplit("<|im_end|>", 1)
                            modified_user_text = parts[0] + f" {structure_tokens[0]}<|im_end|>" + parts[1] if len(parts) > 1 else parts[0] + f" {structure_tokens[0]}"
                        else:
                            # Fallback: append to end
                            modified_user_text = user_text + f" {structure_tokens[0]}"
                        
                        # Re-tokenize with the structure token
                        # Reconstruct full text with modified user part
                        if first_assistant_idx is not None:
                            assistant_messages = messages[last_user_idx + 1:]
                            assistant_text = tokenizer.apply_chat_template(
                                assistant_messages,
                                tokenize=False,
                                add_generation_prompt=False,
                            )
                            full_modified_text = modified_user_text + assistant_text
                        else:
                            full_modified_text = modified_user_text
                        
                        # Re-tokenize
                        modified_tokenized = tokenizer(
                            full_modified_text,
                            truncation=True,
                            max_length=max_seq_length,
                            padding=False,
                        )
                        
                        # Update input_ids, labels, AND attention_mask for this example
                        tokenized["input_ids"][idx] = modified_tokenized["input_ids"]
                        tokenized["labels"][idx] = modified_tokenized["input_ids"][:]
                        if "attention_mask" in tokenized and "attention_mask" in modified_tokenized:
                            tokenized["attention_mask"][idx] = modified_tokenized["attention_mask"]
                        
                        # Find the structure token position in modified sequence
                        structure_token_id = tokenizer.convert_tokens_to_ids(structure_tokens[0])
                        if structure_token_id in modified_tokenized["input_ids"]:
                            patch_pos = modified_tokenized["input_ids"].index(structure_token_id)
            
            patch_position_batch.append(patch_pos)
        tokenized["patch_position"] = patch_position_batch
        
        # Compute instruction positions as a prefix length (system + user turns, up to first assistant).
        # This is robust and avoids brittle substring/offset mapping logic.
        if "instr_positions" not in examples and "messages" in examples:
            instr_len_batch: List[int] = []
            for messages, full_ids in zip(examples["messages"], tokenized["input_ids"]):
                first_assistant_idx = None
                for i, msg in enumerate(messages):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        first_assistant_idx = i
                        break
                prompt_messages = messages if first_assistant_idx is None else messages[:first_assistant_idx]
                add_generation_prompt = first_assistant_idx is not None
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
                prompt_tok = tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                )
                instr_len = min(len(prompt_tok["input_ids"]), len(full_ids))
                instr_len_batch.append(int(instr_len))
            
            tokenized["instr_len"] = instr_len_batch
        
        # Note: We don't need to explicitly preserve other fields here!
        # HuggingFace datasets automatically keeps all columns that aren't in remove_columns.
        # Copying large graph arrays (node_feat, pos, edge_index) would be very slow.
        
        # Override with pre-existing instr_positions/instr_len/patch_position if present
        # (These are small and cheap to copy if they exist)
        if 'instr_positions' in examples:
            tokenized['instr_positions'] = examples['instr_positions']
        if 'instr_len' in examples:
            tokenized['instr_len'] = examples['instr_len']
        if 'patch_position' in examples:
            tokenized['patch_position'] = examples['patch_position']
        
        return tokenized
    
    # Apply preprocessing to dataset - ONLY tokenization, keep raw graph columns
    colnames = dataset[split].column_names
    
    # Check if raw graph columns exist (for logging only)
    has_raw_graph_columns = (
        "modality" in colnames
        and "node_feat" in colnames
        and "pos" in colnames
    )
    if has_raw_graph_columns:
        logger.info("Dataset has raw graph columns - will be used directly by collator during training")
    
    # Optional: Filter dataset based on max_atoms and max_edges BEFORE tokenization
    # WARNING: This can be very slow for large datasets with nested structures.
    # By default, filtering happens during training via the collator's skip_on_error mechanism.
    if filter_before_tokenization and (max_atoms is not None or max_edges is not None) and has_raw_graph_columns:
        original_len = len(dataset[split])
        
        def _filter_by_size(example):
            """Filter out samples that exceed max_atoms or max_edges."""
            try:
                # Count atoms
                num_atoms = 0
                if 'node_feat' in example and example['node_feat'] is not None:
                    num_atoms = len(example['node_feat'])
                
                # Count edges
                num_edges = 0
                if 'edge_index' in example and example['edge_index'] is not None:
                    edge_index = example['edge_index']
                    if isinstance(edge_index, list) and len(edge_index) > 0:
                        num_edges = len(edge_index[0]) if len(edge_index) > 0 else 0
                
                # Additional chemical edges for molecules
                if 'chem_edge_index' in example and example['chem_edge_index'] is not None:
                    chem_edge_index = example['chem_edge_index']
                    if isinstance(chem_edge_index, list) and len(chem_edge_index) > 0:
                        num_edges += len(chem_edge_index[0]) if len(chem_edge_index) > 0 else 0
                
                # Check thresholds
                if max_atoms is not None and num_atoms > max_atoms:
                    return False
                
                if max_edges is not None and num_edges > max_edges:
                    return False
                
                return True
            except Exception as e:
                # Log error but keep the sample to avoid blocking
                logger.warning(f"Error filtering sample: {e}")
                return True
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Filtering dataset based on structure size thresholds:")
        if max_atoms is not None:
            logger.info(f"  Max atoms: {max_atoms:,}")
        if max_edges is not None:
            logger.info(f"  Max edges: {max_edges:,}")
        logger.info(f"Original dataset size: {original_len:,} samples")
        logger.info(f"{'='*80}")
        
        # Use num_proc=1 to avoid multiprocessing issues
        dataset[split] = dataset[split].filter(
            _filter_by_size, 
            desc="Filtering by structure size",
            num_proc=1,
        )
        
        filtered_len = len(dataset[split])
        removed_count = original_len - filtered_len
        logger.info(f"Filtered dataset: {original_len:,} -> {filtered_len:,} samples ({removed_count:,} removed)")
        logger.info(f"{'='*80}\n")
    elif (max_atoms is not None or max_edges is not None) and has_raw_graph_columns:
        logger.info(f"\n{'='*80}")
        logger.info(f"Structure size filtering thresholds configured:")
        if max_atoms is not None:
            logger.info(f"  Max atoms: {max_atoms:,}")
        if max_edges is not None:
            logger.info(f"  Max edges: {max_edges:,}")
        logger.info(f"Filtering will happen during training (via collator skip_on_error)")
        logger.info(f"{'='*80}\n")
    
    # Tokenize dataset - only remove 'messages', keep all graph columns
    remove_cols: List[str] = []
    if "messages" in colnames:
        remove_cols.append("messages")
    
    # Tokenize (fast - only text processing, no graph data copying)
    # Note: num_proc MUST be None for datasets with large nested lists (>1024 elements)
    # PyArrow has a default limit that truncates nested lists during multiprocess serialization

    processed_dataset = dataset.map(
        _preprocess_batch,
        batched=True,
        desc="Tokenizing dataset",
        remove_columns=remove_cols,
        num_proc=None,  # CRITICAL: multiprocessing truncates large nested lists!
    )

    # All raw graph columns (modality, node_feat, pos, edge_index, etc.) are preserved!
    # The collator will build graph_data structures on-the-fly during training.
    logger.info("✓ Preprocessing complete - raw graph columns preserved for on-the-fly processing")
    
    return processed_dataset


@dataclass
class MultiModalInferenceCollator:
    """
    Collates batches with text and graph structures for Octopus inference.
    
    Similar to MultiModalDataCollator but for generation:
    - No labels needed (model generates the response)
    - Supports same graph data formats
    - Preserves reference data for evaluation
    """
    
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    max_instr_positions: int = 32  # Max instruction positions to keep per sample
    structure_tokens: List[str] = None  # List of structure tokens to try
    
    def __post_init__(self):
        """Set default structure tokens if not provided."""
        if self.structure_tokens is None:
            self.structure_tokens = ["<STRUCTURE>"]
    
    @staticmethod
    def _as_tensor(x: Any, *, dtype: torch.dtype) -> torch.Tensor:
        """Convert nested python / numpy structures to torch tensors."""
        if x is None:
            raise ValueError("Graph field is None; cannot convert to tensor.")
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        return torch.tensor(x, dtype=dtype)
    
    @staticmethod
    def _normalize_edge_index(ei: torch.Tensor) -> torch.Tensor:
        """Ensure edge_index shape is [2, E]."""
        if ei.dim() != 2:
            raise ValueError(f"edge_index must be 2D, got shape={tuple(ei.shape)}")
        if ei.size(0) == 2:
            return ei
        if ei.size(1) == 2:
            return ei.t().contiguous()
        raise ValueError(f"edge_index must be [2, E] or [E, 2], got shape={tuple(ei.shape)}")
    
    def _pad_text_features(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pad text sequences and return batch."""
        text_features = []
        
        for f in features:
            text_feat = {
                "input_ids": f["input_ids"],
                "attention_mask": f.get("attention_mask", [1] * len(f["input_ids"])),
            }
            text_features.append(text_feat)
        
        # Use tokenizer to pad input_ids and attention_mask
        padding_strategy = 'longest' if self.padding is True else self.padding
        batch = self.tokenizer.pad(
            text_features,
            padding=padding_strategy,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        return batch
    
    def _collate_graph_data(self, features: List[Dict[str, Any]]) -> Optional[tuple]:
        """Collate graph structures from multiple examples (same as training)."""
        # Check if we have raw graph columns (HF dataset format)
        has_raw_columns = (
            len(features) > 0 and 
            'modality' in features[0] and 
            'node_feat' in features[0] and 
            'pos' in features[0]
        )
        
        if has_raw_columns:
            graph_values = []
            graph_indices = []
            modalities = []
            
            for i, f in enumerate(features):
                if f.get('node_feat') is not None and f.get('pos') is not None:
                    modalities.append(f['modality'])
                    graph_indices.append(i)
                    
                    value: Dict[str, torch.Tensor] = {
                        "node_feat": self._as_tensor(f["node_feat"], dtype=torch.float32),
                        "pos": self._as_tensor(f["pos"], dtype=torch.float32),
                    }
                    
                    if "edge_index" in f and f["edge_index"] is not None:
                        ei = self._as_tensor(f["edge_index"], dtype=torch.long)
                        if ei.numel() > 0:
                            value["edge_index"] = self._normalize_edge_index(ei)
                    
                    if "edge_attr" in f and f["edge_attr"] is not None:
                        value["edge_attr"] = self._as_tensor(f["edge_attr"], dtype=torch.float32)
                    
                    if "edge_feat_dist" in f and f["edge_feat_dist"] is not None:
                        value["edge_feat_dist"] = self._as_tensor(f["edge_feat_dist"], dtype=torch.float32)
                    
                    if "chem_edge_index" in f and f["chem_edge_index"] is not None:
                        cei = self._as_tensor(f["chem_edge_index"], dtype=torch.long)
                        if cei.numel() > 0:
                            value["chem_edge_index"] = self._normalize_edge_index(cei)
                    
                    if "chem_edge_feat_cat" in f and f["chem_edge_feat_cat"] is not None:
                        value["chem_edge_feat_cat"] = self._as_tensor(f["chem_edge_feat_cat"], dtype=torch.long)
                    
                    graph_values.append(value)
            
            if not graph_values:
                return None
            
            if len(set(modalities)) > 1:
                modality_counts = {}
                for m in modalities:
                    modality_counts[m] = modality_counts.get(m, 0) + 1
                raise ValueError(
                    f"Mixed modalities in batch not supported. Found: {modality_counts}. "
                    f"Ensure batches contain only one modality."
                )
            
            modality = modalities[0]
        else:
            # Pre-built graph_data dicts (backward compatibility)
            graphs = []
            graph_indices = []
            for i, f in enumerate(features):
                if 'graph_data' in f and f['graph_data'] is not None:
                    graphs.append(f['graph_data'])
                    graph_indices.append(i)
            
            if not graphs:
                return None
            
            modalities = [g['modality'] for g in graphs]
            if len(set(modalities)) > 1:
                modality_counts = {}
                for m in modalities:
                    modality_counts[m] = modality_counts.get(m, 0) + 1
                raise ValueError(f"Mixed modalities in batch not supported. Found: {modality_counts}")
            
            modality = modalities[0]
            graph_values = [g['value'] for g in graphs]
            
            # Convert to tensors
            normalized_values: List[Dict[str, torch.Tensor]] = []
            for gv in graph_values:
                if not isinstance(gv, dict):
                    raise ValueError(f"graph_data['value'] must be a dict, got {type(gv)}")
                
                out: Dict[str, torch.Tensor] = {}
                out["node_feat"] = self._as_tensor(gv["node_feat"], dtype=torch.float32)
                out["pos"] = self._as_tensor(gv["pos"], dtype=torch.float32)
                
                if "edge_index" in gv and gv["edge_index"] is not None:
                    ei = self._as_tensor(gv["edge_index"], dtype=torch.long)
                    out["edge_index"] = self._normalize_edge_index(ei)
                
                if "edge_attr" in gv and gv["edge_attr"] is not None:
                    out["edge_attr"] = self._as_tensor(gv["edge_attr"], dtype=torch.float32)
                
                if "edge_feat_dist" in gv and gv["edge_feat_dist"] is not None:
                    out["edge_feat_dist"] = self._as_tensor(gv["edge_feat_dist"], dtype=torch.float32)
                
                if "chem_edge_index" in gv and gv["chem_edge_index"] is not None:
                    cei = self._as_tensor(gv["chem_edge_index"], dtype=torch.long)
                    out["chem_edge_index"] = self._normalize_edge_index(cei)
                
                if "chem_edge_feat_cat" in gv and gv["chem_edge_feat_cat"] is not None:
                    out["chem_edge_feat_cat"] = self._as_tensor(gv["chem_edge_feat_cat"], dtype=torch.long)
                
                normalized_values.append(out)
            
            graph_values = normalized_values
        
        # Merge graphs using modality-specific logic
        if modality == 'protein':
            if any("edge_index" not in gv for gv in graph_values):
                raise ValueError("Protein modality requires `edge_index` in graph_data['value'].")
            merged = merge_protein_graphs(graph_values)
        else:  # molecule, DNA, RNA
            merged = merge_molecule_graphs(graph_values)
        
        batched_graph = {
            'modality': modality,
            'value': merged,
        }
        
        batch_tensor = merged['batch']
        
        return batched_graph, batch_tensor, graph_indices
    
    def _add_instr_positions(self, batch: Dict[str, Any], features: List[Dict[str, Any]]) -> None:
        """Add padded instruction positions to batch."""
        if not any(('instr_positions' in f) or ('instr_len' in f) for f in features):
            return
        
        instr_pos_list = []
        for f in features:
            if 'instr_positions' in f:
                pos = f.get('instr_positions', [-1])
            else:
                instr_len = int(f.get('instr_len', 0) or 0)
                pos = list(range(min(instr_len, self.max_instr_positions))) if instr_len > 0 else [-1]
            pos = pos[:self.max_instr_positions]
            instr_pos_list.append(pos)
        
        max_len = max(len(p) for p in instr_pos_list)
        
        padded_positions = []
        for pos in instr_pos_list:
            padded = pos + [-1] * (max_len - len(pos))
            padded_positions.append(padded)
        
        batch['instr_positions'] = torch.tensor(padded_positions, dtype=torch.long)
    
    def _add_patch_positions(self, batch: Dict[str, Any], features: List[Dict[str, Any]]) -> None:
        """Add patch positions to batch."""
        if not any('patch_position' in f for f in features):
            return
        
        patch_pos_list = []
        for f in features:
            pos = f.get('patch_position', -1)
            patch_pos_list.append(pos)
        
        batch['patch_positions'] = torch.tensor(patch_pos_list, dtype=torch.long).unsqueeze(-1)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of examples for inference.
        
        Returns batch compatible with Octopus.forward():
        - input_ids: [B, seq_len]
        - attention_mask: [B, seq_len]
        - graph_data: {'modality': str, 'value': {...}} (if graphs present)
        - batch: [N] node-to-graph assignment (if graphs present)
        - instr_positions: [B, max_instr_len] (if present)
        - patch_positions: [B, 1] (if present)
        
        Also preserves reference data for evaluation:
        - reference_text: List of expected responses (if present)
        - sample_ids: List of sample identifiers (if present)
        """
        # Pad text features (input_ids, attention_mask)
        batch = self._pad_text_features(features)
        
        # Collate graph data if present
        graph_result = self._collate_graph_data(features)
        if graph_result is not None:
            graph_data, batch_tensor, graph_indices = graph_result
            batch['graph_data'] = graph_data
            batch['batch'] = batch_tensor
            batch['_graph_indices'] = graph_indices
        
        # Add instruction positions
        self._add_instr_positions(batch, features)
        
        # Add patch positions
        self._add_patch_positions(batch, features)
        
        # Preserve reference data for evaluation (not used in forward pass)
        if 'reference_text' in features[0]:
            batch['reference_text'] = [f.get('reference_text', '') for f in features]
        
        if 'sample_id' in features[0]:
            batch['sample_id'] = [f.get('sample_id', '') for f in features]
        
        return batch


class ModalityAwareBatchSamplerForSFT(Sampler):
    """
    Batch sampler that ensures all samples in a batch have the same modality.
    
    This is required for multimodal SFT training because the collator doesn't support
    mixed modalities in a single batch. This sampler groups samples by modality and
    creates batches within each modality group.
    
    Optimized for HuggingFace datasets with either:
    - Direct 'modality' column (fast path)
    - Nested 'graph_data' column containing modality (slow path)
    
    Args:
        dataset: HuggingFace Dataset instance with multimodal data
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle samples within each modality group
        seed: Random seed for shuffling
        drop_last: Whether to drop the last incomplete batch
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        
        # Group indices by modality
        self.modality_indices = {}
        print("\nGrouping SFT dataset by modality...")
        
        dataset_len = len(dataset)
        
        # Check if modality column exists directly (fast path)
        if 'modality' in dataset.column_names:
            # FAST PATH: Bulk read all modalities at once
            print("  Using direct 'modality' column (fast path)")
            modalities = dataset['modality']
            
            for idx, modality in enumerate(modalities):
                if modality not in self.modality_indices:
                    self.modality_indices[modality] = []
                self.modality_indices[modality].append(idx)
        
        # Check if graph_data column exists (slower path)
        elif 'graph_data' in dataset.column_names:
            print("  Extracting modality from 'graph_data' column (slow path)")
            graph_data_list = dataset['graph_data']
            
            for idx, graph_data in enumerate(graph_data_list):
                modality = ''
                if graph_data is not None and isinstance(graph_data, dict):
                    modality = graph_data.get('modality', '')
                
                if modality not in self.modality_indices:
                    self.modality_indices[modality] = []
                self.modality_indices[modality].append(idx)
        
        else:
            # No modality information - create single group
            print("  Warning: No modality column found, treating all samples as same modality")
            self.modality_indices['unknown'] = list(range(dataset_len))
        
        # Log modality distribution
        print("-" * 80)
        total_samples = 0
        for modality, indices in sorted(self.modality_indices.items()):
            print(f"  {modality}: {len(indices):,} samples")
            total_samples += len(indices)
        print(f"  Total: {total_samples:,} samples")
        print("-" * 80 + "\n")
        
        self.epoch = 0
    
    def __iter__(self):
        # Set random seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Shuffle indices within each modality if needed
        all_batches = []
        for modality, indices in self.modality_indices.items():
            indices_copy = indices.copy()
            
            if self.shuffle:
                # Shuffle within modality
                random.Random(self.seed + self.epoch).shuffle(indices_copy)
            
            # Create batches for this modality
            for i in range(0, len(indices_copy), self.batch_size):
                batch = indices_copy[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle the order of batches (so different modalities are interleaved)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        total_batches = 0
        for indices in self.modality_indices.values():
            num_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                num_batches += 1
            total_batches += num_batches
        return total_batches
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch


def preprocess_inference_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    max_seq_length: int = 2048,
    structure_tokens: List[str] = None,
    insert_structure_if_missing: bool = True,
) -> DatasetDict:
    """
    Preprocess multimodal dataset for inference by tokenizing prompts only.
    
    Similar to preprocess_multimodal_dataset but:
    1. Removes the last assistant message from messages
    2. Tokenizes only the prompt (system + user messages)
    3. No labels are created (model will generate the response)
    4. Preserves the last assistant message as reference_text for evaluation
    5. Finds structure token positions for patch injection
    6. Computes instruction positions (full prompt)
    7. Preserves graph_data and other fields from dataset
    
    Args:
        dataset: Dataset dictionary containing the data
        tokenizer: Tokenizer to use for processing
        split: Split name to process (e.g., "test")
        max_seq_length: Maximum sequence length for truncation
        structure_tokens: List of structure tokens to search for
        insert_structure_if_missing: If True, insert structure token at end of user query when not found
    
    Returns:
        Processed dataset with tokenized prompts, patch_position, and reference texts
    """
    if structure_tokens is None:
        structure_tokens = ["<STRUCTURE>"]
    
    def _preprocess_batch(examples):
        """Tokenize prompts and prepare for inference while preserving multimodal fields."""
        # Check if examples have 'messages' or if they're already tokenized
        if 'messages' not in examples and 'input_ids' in examples:
            # Already preprocessed, just return
            return examples
        
        # Process each example to extract prompt and reference
        prompts = []
        references = []
        sample_ids = []
        
        for idx, messages in enumerate(examples["messages"]):
            # Find last assistant message for reference
            last_assistant_msg = None
            prompt_messages = []
            
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("role") == "assistant":
                        last_assistant_msg = msg.get("content", "")
                    else:
                        prompt_messages.append(msg)
            
            # If all messages including assistant, remove last assistant
            if last_assistant_msg is None:
                # No assistant message found, use all messages as prompt
                prompt_messages = messages
                last_assistant_msg = ""
            
            # Build prompt with generation prompt
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,  # Add assistant header for generation
            )
            
            prompts.append(prompt_text)
            references.append(last_assistant_msg)
            
            # Create sample ID if available
            if 'id' in examples:
                sample_ids.append(examples['id'][idx])
            elif 'sample_id' in examples:
                sample_ids.append(examples['sample_id'][idx])
            else:
                sample_ids.append(f"sample_{idx}")
        
        # Tokenize prompts
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # Padding handled by collator
        )
        
        # Find <STRUCTURE> token position for patch injection
        patch_position_batch: List[int] = []
        for idx, input_ids in enumerate(tokenized["input_ids"]):
            # Try each structure token in the list
            patch_pos = -1
            for structure_token in structure_tokens:
                structure_token_id = tokenizer.convert_tokens_to_ids(structure_token)
                if structure_token_id is not None and structure_token_id in input_ids:
                    patch_pos = input_ids.index(structure_token_id)
                    break
            
            # Fallback: Insert structure token at end of user query if not found
            # BUT ONLY if this sample actually has graph data!
            if patch_pos == -1 and insert_structure_if_missing:
                # Check if this sample has graph data
                has_graph = False
                if 'node_feat' in examples and idx < len(examples['node_feat']):
                    # Raw column format
                    has_graph = examples['node_feat'][idx] is not None
                elif 'graph_data' in examples and idx < len(examples['graph_data']):
                    # Nested graph_data format
                    has_graph = examples['graph_data'][idx] is not None
                
                # Only insert structure token if graph data exists
                if has_graph:
                    messages = examples["messages"][idx]
                    
                    # Find last user message
                    last_user_idx = None
                    for i, msg in enumerate(messages):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            last_user_idx = i
                    
                    if last_user_idx is not None:
                        # Reconstruct prompt messages with structure token
                        prompt_messages = [msg for msg in messages if msg.get("role") != "assistant"]
                        
                        # Get last user message text and insert structure token
                        user_text = tokenizer.apply_chat_template(
                            prompt_messages[:last_user_idx + 1],
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                        
                        # Insert structure token before closing tag
                        if "<|im_end|>" in user_text:
                            parts = user_text.rsplit("<|im_end|>", 1)
                            modified_user_text = parts[0] + f" {structure_tokens[0]}<|im_end|>" + (parts[1] if len(parts) > 1 else "")
                        else:
                            modified_user_text = user_text + f" {structure_tokens[0]}"
                        
                        # Complete prompt with generation prompt
                        if last_user_idx < len(prompt_messages) - 1:
                            remaining_messages = prompt_messages[last_user_idx + 1:]
                            remaining_text = tokenizer.apply_chat_template(
                                remaining_messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            full_modified_text = modified_user_text + remaining_text
                        else:
                            # Add generation prompt to modified text
                            full_modified_text = modified_user_text
                            if not full_modified_text.endswith("<|im_start|>assistant\n"):
                                full_modified_text += "<|im_start|>assistant\n"
                        
                        # Re-tokenize
                        modified_tokenized = tokenizer(
                            full_modified_text,
                            truncation=True,
                            max_length=max_seq_length,
                            padding=False,
                        )
                        
                        # Update input_ids and attention_mask for this example
                        tokenized["input_ids"][idx] = modified_tokenized["input_ids"]
                        if "attention_mask" in tokenized and "attention_mask" in modified_tokenized:
                            tokenized["attention_mask"][idx] = modified_tokenized["attention_mask"]
                        
                        # Find the structure token position
                        structure_token_id = tokenizer.convert_tokens_to_ids(structure_tokens[0])
                        if structure_token_id in modified_tokenized["input_ids"]:
                            patch_pos = modified_tokenized["input_ids"].index(structure_token_id)
            
            patch_position_batch.append(patch_pos)
        
        tokenized["patch_position"] = patch_position_batch
        
        # For inference, instruction positions are the entire prompt
        instr_len_batch: List[int] = []
        for full_ids in tokenized["input_ids"]:
            instr_len = len(full_ids)
            instr_len_batch.append(int(instr_len))
        
        tokenized["instr_len"] = instr_len_batch
        
        # Store reference texts for evaluation
        tokenized["reference_text"] = references
        tokenized["sample_id"] = sample_ids
        
        # Override with pre-existing fields if present
        if 'instr_positions' in examples:
            tokenized['instr_positions'] = examples['instr_positions']
        if 'instr_len' in examples:
            tokenized['instr_len'] = examples['instr_len']
        if 'patch_position' in examples:
            tokenized['patch_position'] = examples['patch_position']
        
        return tokenized
    
    # Apply preprocessing to dataset - ONLY tokenization, keep raw graph columns
    colnames = dataset[split].column_names
    
    # Check if raw graph columns exist (for logging only)
    has_raw_graph_columns = (
        "modality" in colnames
        and "node_feat" in colnames
        and "pos" in colnames
    )
    if has_raw_graph_columns:
        logger.info("Dataset has raw graph columns - will be used directly by collator during inference")
    
    # Tokenize dataset - only remove 'messages', keep all graph columns
    remove_cols: List[str] = []
    if "messages" in colnames:
        remove_cols.append("messages")
    
    # Tokenize (fast - only text processing, no graph data copying)
    processed_dataset = dataset.map(
        _preprocess_batch,
        batched=True,
        desc="Tokenizing inference dataset",
        remove_columns=remove_cols,
        num_proc=None,  # CRITICAL: multiprocessing truncates large nested lists!
    )
    
    logger.info("✓ Inference preprocessing complete - raw graph columns preserved")
    
    return processed_dataset
