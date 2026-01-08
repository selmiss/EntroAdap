import argparse
import json
import os
from collections import Counter
from typing import Iterable, Optional

import torch
from safetensors import safe_open


def _read_index_keys(index_path: str) -> tuple[list[str], list[str]]:
    """
    Read a huggingface index file and return (keys, shard_files).
    """
    with open(index_path, "r") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    keys = list(weight_map.keys())
    shards = sorted(set(weight_map.values()))
    return keys, shards


def _list_safetensor_keys(file_path: str) -> list[str]:
    """
    Return tensor keys from a safetensors file without loading tensor data.
    """
    with safe_open(file_path, framework="pt", device="meta") as f:
        return list(f.keys())


def _list_pytorch_bin_keys(file_path: str) -> list[str]:
    """
    Return tensor keys from a PyTorch bin file. Uses meta device when possible
    to avoid host memory usage; falls back to CPU if unsupported.
    """
    try:
        state_dict = torch.load(file_path, map_location="meta")
    except Exception:
        state_dict = torch.load(file_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return list(state_dict.keys())


def _group_prefixes(keys: Iterable[str], depth: int) -> Counter:
    def _prefix(key: str) -> str:
        parts = key.split(".")
        return ".".join(parts[:depth]) if len(parts) >= depth else key

    return Counter(_prefix(k) for k in keys)


def inspect_checkpoint_keys(
    checkpoint_path: str,
    limit: Optional[int] = None,
    prefix_depth: int = 5,
    sort_keys: bool = True,
    quiet: bool = False,
) -> dict:
    """
    Inspect checkpoint tensor keys without loading full weights when possible.

    Args:
        checkpoint_path: Directory containing weights or a direct file path.
        limit: Number of keys to show in the sample output.
        prefix_depth: How many dot-delimited segments to aggregate for prefix counts.
        sort_keys: Whether to sort keys before sampling.
        quiet: If True, suppress stdout logging and only return the summary dict.

    Returns:
        Dict with keys: path, total_keys, sample_keys, prefix_counts, shard_files.
    """
    path = os.path.expanduser(checkpoint_path)
    shard_files: list[str] = []
    keys: list[str] = []
    source: str | None = None

    if os.path.isdir(path):
        safetensor_index = os.path.join(path, "model.safetensors.index.json")
        torch_index = os.path.join(path, "pytorch_model.bin.index.json")

        if os.path.exists(safetensor_index):
            keys, shard_files = _read_index_keys(safetensor_index)
            source = "safetensors index"
        elif os.path.exists(torch_index):
            keys, shard_files = _read_index_keys(torch_index)
            source = "pytorch bin index"
        else:
            candidate_files = [
                ("model.safetensors", _list_safetensor_keys, "safetensors"),
                ("pytorch_model.bin", _list_pytorch_bin_keys, "pytorch bin"),
            ]
            for filename, loader, src in candidate_files:
                full = os.path.join(path, filename)
                if os.path.exists(full):
                    keys = loader(full)
                    shard_files = [filename]
                    source = src
                    break

    else:
        if path.endswith(".safetensors"):
            keys = _list_safetensor_keys(path)
            shard_files = [os.path.basename(path)]
            source = "safetensors"
        elif path.endswith(".bin"):
            keys = _list_pytorch_bin_keys(path)
            shard_files = [os.path.basename(path)]
            source = "pytorch bin"

    if not keys:
        raise FileNotFoundError(
            f"Could not find checkpoint weights or index at {checkpoint_path}"
        )

    if sort_keys:
        keys = sorted(keys)

    prefix_counts = _group_prefixes(keys, max(1, prefix_depth))
    sample = keys[:limit] if limit > 0 else keys

    summary = {
        "path": os.path.abspath(path),
        "source": source,
        "total_keys": len(keys),
        "sample_keys": sample,
        "prefix_counts": prefix_counts.most_common(),
        "shard_files": shard_files,
    }

    if not quiet:
        print(f"Checkpoint: {summary['path']}")
        print(f"Source: {summary['source']}")
        if shard_files:
            print(f"Shard files: {', '.join(shard_files)}")
        print(f"Total tensor keys: {summary['total_keys']}")

        print("\nTop prefixes:")
        for prefix, count in summary["prefix_counts"][:20]:
            print(f"  {prefix}: {count}")

        if sample:
            print(f"\nSample keys (first {len(sample)}):")
            for key in sample:
                print(f"  - {key}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint tensor keys without loading full weights."
    )
    parser.add_argument(
        "checkpoint_path",
        help="Path to checkpoint directory or weight file (.safetensors / .bin).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of keys to display in the sample list.",
    )
    parser.add_argument(
        "--prefix_depth",
        type=int,
        default=5,
        help="Dot-delimited depth to group prefix counts.",
    )
    parser.add_argument(
        "--no_sort",
        action="store_true",
        help="Do not sort keys before sampling (keeps original order).",
    )

    args = parser.parse_args()
    inspect_checkpoint_keys(
        checkpoint_path=args.checkpoint_path,
        limit=args.limit,
        prefix_depth=args.prefix_depth,
        sort_keys=not args.no_sort,
    )


if __name__ == "__main__":
    main()

