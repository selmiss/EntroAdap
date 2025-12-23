#!/usr/bin/env python3
# coding=utf-8
"""
Tests for multi-modal LLM model.

NOTE: This test file is maintained for API reference of the OLD interface.
For the NEW integrated architecture with encoder and patching, see:
    tests/test_octopus_integrated.py

The Octopus class has been refactored to integrate:
- AAEncoder (graph encoder)
- Instruction-conditioned patching (con_gates)
- Cross-attention fusion (patches as Q, nodes as KV)

Old interface (deprecated):
    - modality_embeddings: pre-computed embeddings
    - kv_embeddings: pre-computed KV embeddings
    
New interface:
    - graph_data: raw graph data
    - batch: node-to-graph assignment
    - instr_positions: token positions containing instructions
    - patch_positions: positions to inject patches
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from src.models.octopus import Octopus


@pytest.mark.skip(reason="Old API deprecated - see test_octopus_integrated.py for new tests")
class TestOctopus:
    """Tests for Octopus (OLD API - DEPRECATED)."""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
