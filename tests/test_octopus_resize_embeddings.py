#!/usr/bin/env python3
# coding=utf-8
"""
Unit test that Octopus delegates resize_token_embeddings() to its LLM backbone.
"""

import pytest

from transformers import GPT2Config, GPT2LMHeadModel

from src.models.octopus import Octopus
from src.models.octopus_config import BaseConfig


class TestOctopusResizeEmbeddings:
    @pytest.mark.unit
    def test_resize_token_embeddings_resizes_backbone(self):
        cfg = GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
        )
        llm = GPT2LMHeadModel(cfg)
        mm = Octopus(llm_model=llm, config=BaseConfig())

        mm.resize_token_embeddings(40)

        assert mm.llm_model.get_input_embeddings().weight.shape[0] == 40
        # Delegation should make wrapper accessors consistent too
        assert mm.get_input_embeddings().weight.shape[0] == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


