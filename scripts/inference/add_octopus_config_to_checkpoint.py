#!/usr/bin/env python3
"""
Utility script to add octopus_config to an existing checkpoint's config.json.

This helps ensure proper model loading during inference by storing the
Octopus architecture parameters in the checkpoint.

Usage:
    python scripts/inference/add_octopus_config_to_checkpoint.py \
        --checkpoint_path ./checkpoints/octopus/model \
        --encoder_hidden_dim 256 \
        --num_fusion_blocks 8 \
        --fusion_hidden_dim 4096
"""

import json
import os
import argparse
from pathlib import Path


def add_octopus_config(
    checkpoint_path: str,
    encoder_hidden_dim: int = 256,
    encoder_num_layers: int = 6,
    encoder_dropout: float = 0.1,
    patching_k_max: int = 32,
    patching_r_max: int = 64,
    fusion_num_blocks: int = 8,
    fusion_num_heads: int = 14,
    fusion_hidden_dim: int = 896,
    fusion_intermediate_dim: int = None,
    fusion_dropout: float = 0.1,
    backup: bool = True,
):
    """
    Add octopus_config to checkpoint's config.json.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        encoder_hidden_dim: Encoder hidden dimension
        encoder_num_layers: Number of encoder layers
        encoder_dropout: Encoder dropout rate
        patching_k_max: Maximum number of patches
        patching_r_max: Maximum radius for patching
        fusion_num_blocks: Number of fusion blocks
        fusion_num_heads: Number of attention heads in fusion
        fusion_hidden_dim: Fusion hidden dimension
        fusion_intermediate_dim: Fusion intermediate dimension (default: 4 * hidden_dim)
        fusion_dropout: Fusion dropout rate
        backup: Whether to create a backup of config.json
    """
    config_path = os.path.join(checkpoint_path, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create backup if requested
    if backup:
        backup_path = config_path + ".backup"
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Created backup: {backup_path}")
    
    # Check if it's an Octopus model
    if "architectures" not in config or "Octopus" not in config["architectures"]:
        print("Warning: This doesn't appear to be an Octopus model (architectures field)")
        print("Proceeding anyway...")
    
    # Set fusion_intermediate_dim default
    if fusion_intermediate_dim is None:
        fusion_intermediate_dim = 4 * fusion_hidden_dim
    
    # Create octopus_config
    octopus_config = {
        "encoder": {
            "hidden_dim": encoder_hidden_dim,
            "num_layers": encoder_num_layers,
            "dropout": encoder_dropout,
        },
        "patching": {
            "k_max": patching_k_max,
            "r_max": patching_r_max,
        },
        "fusion": {
            "num_blocks": fusion_num_blocks,
            "num_heads": fusion_num_heads,
            "hidden_dim": fusion_hidden_dim,
            "intermediate_dim": fusion_intermediate_dim,
            "dropout": fusion_dropout,
        }
    }
    
    # Add to config
    config["octopus_config"] = octopus_config
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Added octopus_config to {config_path}")
    print(f"\nConfiguration:")
    print(f"  Encoder: hidden_dim={encoder_hidden_dim}, num_layers={encoder_num_layers}")
    print(f"  Fusion: num_blocks={fusion_num_blocks}, hidden_dim={fusion_hidden_dim}")
    print(f"  Patching: k_max={patching_k_max}, r_max={patching_r_max}")


def main():
    parser = argparse.ArgumentParser(
        description="Add octopus_config to checkpoint's config.json"
    )
    
    # Required
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    
    # Encoder config
    parser.add_argument(
        "--encoder_hidden_dim",
        type=int,
        default=256,
        help="Encoder hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--encoder_num_layers",
        type=int,
        default=6,
        help="Number of encoder layers (default: 6)",
    )
    parser.add_argument(
        "--encoder_dropout",
        type=float,
        default=0.1,
        help="Encoder dropout rate (default: 0.1)",
    )
    
    # Patching config
    parser.add_argument(
        "--patching_k_max",
        type=int,
        default=32,
        help="Maximum number of patches (default: 32)",
    )
    parser.add_argument(
        "--patching_r_max",
        type=int,
        default=64,
        help="Maximum radius for patching (default: 64)",
    )
    
    # Fusion config
    parser.add_argument(
        "--fusion_num_blocks",
        type=int,
        default=8,
        help="Number of fusion blocks (default: 8)",
    )
    parser.add_argument(
        "--fusion_num_heads",
        type=int,
        default=14,
        help="Number of attention heads in fusion (default: 14)",
    )
    parser.add_argument(
        "--fusion_hidden_dim",
        type=int,
        default=896,
        help="Fusion hidden dimension (default: 896 for Qwen2-0.5B, 4096 for Llama-3.1-8B)",
    )
    parser.add_argument(
        "--fusion_intermediate_dim",
        type=int,
        default=None,
        help="Fusion intermediate dimension (default: 4 * fusion_hidden_dim)",
    )
    parser.add_argument(
        "--fusion_dropout",
        type=float,
        default=0.1,
        help="Fusion dropout rate (default: 0.1)",
    )
    
    # Other
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Don't create backup of config.json",
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.isdir(args.checkpoint_path):
        print(f"Error: Checkpoint path does not exist: {args.checkpoint_path}")
        return 1
    
    # Add config
    try:
        add_octopus_config(
            checkpoint_path=args.checkpoint_path,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_num_layers=args.encoder_num_layers,
            encoder_dropout=args.encoder_dropout,
            patching_k_max=args.patching_k_max,
            patching_r_max=args.patching_r_max,
            fusion_num_blocks=args.fusion_num_blocks,
            fusion_num_heads=args.fusion_num_heads,
            fusion_hidden_dim=args.fusion_hidden_dim,
            fusion_intermediate_dim=args.fusion_intermediate_dim,
            fusion_dropout=args.fusion_dropout,
            backup=not args.no_backup,
        )
        print("\n✓ Success! Config updated.")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

