#!/usr/bin/env python3
# coding=utf-8
"""
Unit tests for utils.env_utils.expand_env_vars.
"""

import os

import pytest

from src.configs import ScriptArguments
from utils.env_utils import expand_env_vars


class TestEnvUtils:
    @pytest.mark.unit
    def test_expands_nested_dataset_mixture_dataclasses(self):
        os.environ["DATA_DIR"] = "/tmp/data_dir"
        try:
            args = ScriptArguments(
                dataset_name=None,
                dataset_mixture={
                    "seed": 0,
                    "datasets": [
                        {
                            "id": "${DATA_DIR}/a.jsonl",
                            "split": "train",
                            "weight": 1.0,
                            "columns": ["messages"],
                        },
                        {
                            "id": "${DATA_DIR}/b.jsonl",
                            "split": "train",
                            "weight": 2.0,
                            "columns": ["messages"],
                        },
                    ],
                },
            )

            # dataset_mixture is converted to DatasetMixtureConfig during __post_init__
            expand_env_vars(args)

            assert args.dataset_mixture.datasets[0].id == "/tmp/data_dir/a.jsonl"
            assert args.dataset_mixture.datasets[1].id == "/tmp/data_dir/b.jsonl"
        finally:
            del os.environ["DATA_DIR"]

    @pytest.mark.unit
    def test_expands_dict_keys_and_values(self):
        os.environ["K"] = "hello"
        os.environ["V"] = "world"
        try:
            obj = {"${K}": "${V}"}
            expand_env_vars(obj)
            assert obj == {"hello": "world"}
        finally:
            del os.environ["K"]
            del os.environ["V"]

    @pytest.mark.unit
    def test_cycle_safe(self):
        os.environ["X"] = "x"
        try:
            lst = ["${X}"]
            lst.append(lst)  # cycle
            expand_env_vars(lst)
            assert lst[0] == "x"
            assert lst[1] is lst
        finally:
            del os.environ["X"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


