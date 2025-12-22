# Multi-modal data collator for training with text and graph structures

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy
from datasets import Dataset, DatasetDict

from .graph_batch_utils import merge_protein_graphs, merge_molecule_graphs


@dataclass
class MultiModalDataCollator:
    """
    Collates batches with text and graph structures for MultiModalLLM training.
    
    Works with SFTTrainer by handling pre-tokenized inputs and batching graph data.
    Produces outputs compatible with MultiModalLLM.forward() signature:
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
        batch = self.tokenizer.pad(
            text_features,
            padding=self.padding,
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
        
        Returns:
            (graph_data, batch_tensor, graph_indices) or None if no graphs present
            - graph_data: {'modality': str, 'value': {...}} batched graph
            - batch_tensor: [N] node-to-graph assignment
            - graph_indices: List[int] indices of which examples have graphs
        """
        # Extract graphs from features
        graphs = []
        graph_indices = []
        for i, f in enumerate(features):
            if 'graph_data' in f and f['graph_data'] is not None:
                graphs.append(f['graph_data'])
                graph_indices.append(i)
        
        if not graphs:
            return None
        
        # Group by modality (require all same modality in batch for now)
        modalities = [g['modality'] for g in graphs]
        if len(set(modalities)) > 1:
            raise ValueError(f"Mixed modalities in batch not supported yet: {set(modalities)}")
        
        modality = modalities[0]
        graph_values = [g['value'] for g in graphs]
        
        # Convert common graph fields to tensors (HF datasets typically yield python lists)
        normalized_values: List[Dict[str, torch.Tensor]] = []
        for gv in graph_values:
            if not isinstance(gv, dict):
                raise ValueError(f"graph_data['value'] must be a dict, got {type(gv)}")
            
            out: Dict[str, torch.Tensor] = {}
            
            # Required fields
            out["node_feat"] = self._as_tensor(gv["node_feat"], dtype=torch.float32)
            out["pos"] = self._as_tensor(gv["pos"], dtype=torch.float32)
            
            if "edge_index" in gv and gv["edge_index"] is not None:
                ei = self._as_tensor(gv["edge_index"], dtype=torch.long)
                out["edge_index"] = self._normalize_edge_index(ei)
            
            # Optional fields (molecule/protein)
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
        
        # Merge graphs using same logic as GraphBatchCollator
        if modality == 'protein':
            if any("edge_index" not in gv for gv in normalized_values):
                raise ValueError("Protein modality requires `edge_index` in graph_data['value'].")
            merged = merge_protein_graphs(normalized_values)
        else:  # molecule
            merged = merge_molecule_graphs(normalized_values)
        
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
        
        Returns batch compatible with MultiModalLLM.forward():
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
    max_seq_length: int = 1024,
) -> DatasetDict:
    """
    Preprocess multimodal dataset by tokenizing messages and creating labels.
    
    This function:
    1. Applies chat template to convert messages to text
    2. Tokenizes the text into input_ids
    3. Creates labels (copies of input_ids for causal LM training)
    4. Finds structure token positions for patch injection (supports custom tokens)
    5. Computes instruction positions (full prompt: system + user messages before first assistant response)
    6. Preserves graph_data and other fields from dataset
    
    Args:
        dataset: Dataset dictionary containing the data
        tokenizer: Tokenizer to use for processing (must have structure token registered)
        split: Split name to process (e.g., "train")
        max_seq_length: Maximum sequence length for truncation
    
    Returns:
        Processed dataset with tokenized inputs, labels, patch_position, and instr_len
    """
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
        # Support custom structure tokens per example (default: '<STRUCTURE>')
        patch_position_batch: List[int] = []
        for idx, input_ids in enumerate(tokenized["input_ids"]):
            # Get structure token for this example (default to '<STRUCTURE>')
            if 'structure_token' in examples:
                structure_token = examples['structure_token'][idx] if isinstance(examples['structure_token'], list) else examples['structure_token']
            else:
                structure_token = '<STRUCTURE>'
            
            structure_token_id = tokenizer.convert_tokens_to_ids(structure_token)
            
            # Find first occurrence of structure token
            patch_pos = -1
            if structure_token_id is not None and structure_token_id in input_ids:
                patch_pos = input_ids.index(structure_token_id)
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
        
        # Preserve graph_data if present
        if 'graph_data' in examples:
            tokenized['graph_data'] = examples['graph_data']
        # Or build graph_data from common combined-parquet columns
        elif "modality" in examples and "node_feat" in examples and "pos" in examples:
            graph_data_list: List[Dict[str, Any]] = []
            batch_size = len(examples["modality"])
            for i in range(batch_size):
                value: Dict[str, Any] = {
                    "node_feat": examples["node_feat"][i],
                    "pos": examples["pos"][i],
                }
                # Optional / modality-specific columns
                for k in [
                    "edge_index",
                    "edge_feat_dist",
                    "chem_edge_index",
                    "chem_edge_feat_cat",
                    "edge_attr",
                ]:
                    if k in examples and examples[k][i] is not None:
                        value[k] = examples[k][i]
                graph_data_list.append({"modality": examples["modality"][i], "value": value})
            tokenized["graph_data"] = graph_data_list
        
        # Preserve instr_positions/instr_len if present (overrides computed)
        if 'instr_positions' in examples:
            tokenized['instr_positions'] = examples['instr_positions']
        if 'instr_len' in examples:
            tokenized['instr_len'] = examples['instr_len']
        if 'patch_position' in examples:
            tokenized['patch_position'] = examples['patch_position']
        
        return tokenized
    
    # Apply preprocessing to dataset
    colnames = dataset[split].column_names
    will_build_graph_data = (
        "graph_data" not in colnames
        and "modality" in colnames
        and "node_feat" in colnames
        and "pos" in colnames
    )
    remove_cols: List[str] = []
    if "messages" in colnames:
        remove_cols.append("messages")
    if will_build_graph_data:
        # drop raw graph columns once wrapped into graph_data
        for k in [
            "modality",
            "node_feat",
            "pos",
            "edge_index",
            "edge_feat_dist",
            "chem_edge_index",
            "chem_edge_feat_cat",
            "edge_attr",
        ]:
            if k in colnames:
                remove_cols.append(k)

    processed_dataset = dataset.map(
        _preprocess_batch,
        batched=True,
        desc="Tokenizing dataset",
        remove_columns=remove_cols,
    )
    
    return processed_dataset
