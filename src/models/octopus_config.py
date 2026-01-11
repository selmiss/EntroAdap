"""
Configuration classes for Octopus.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EncoderConfig:
    """Configuration for AAEncoder."""
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    update_coords: bool = True


@dataclass
class PatchingConfig:
    """Configuration for instruction-conditioned patching."""
    k_max: int = 256                          # Max patches per graph
    r_max: int = 1024                          # Max nodes per patch
    steps: int = 7                           # Patch growth iterations
    keep_ratio: float = 0.92                  # Membership retention per step
    dynamic_k_mass: Optional[float] = 0.1   # Mass-based anchor selection (e.g., 0.8)
    gate_hidden_dim: int = 256               # Gate MLP hidden dimension
    gate_dropout: float = 0.0                # Gate dropout


@dataclass
class FusionConfig:
    """Configuration for cross-attention fusion."""
    num_blocks: int = 4                      # Number of fusion blocks
    num_heads: int = 8                       # Attention heads
    hidden_dim: Optional[int] = None         # Fusion hidden dim (None = encoder_hidden_dim)
    intermediate_dim: Optional[int] = None   # FFN intermediate dim (None = 4*hidden_dim)
    dropout: float = 0.1                     # Fusion dropout


@dataclass
class PredictionHeadConfig:
    """Configuration for optional prediction head (regression/classification)."""
    task_type: Optional[str] = None          # None, 'regression', or 'classification'
    num_labels: int = 2                      # Number of labels for classification
    pooling_strategy: str = 'last'           # 'last', 'mean', or 'attention'
    hidden_dim: Optional[int] = None         # Intermediate hidden dim (None = single layer)
    dropout: float = 0.1                     # Dropout rate
    use_dual_loss: bool = False              # Use both LM loss and head loss
    lm_loss_weight: float = 0.5              # Weight for LM loss in dual setup (0.0-1.0)


@dataclass
class OctopusConfig:
    """
    Complete configuration for Octopus.
    
    Example:
        config = OctopusConfig(
            encoder=EncoderConfig(hidden_dim=256, num_layers=6),
            patching=PatchingConfig(k_max=32, r_max=64),
            fusion=FusionConfig(num_blocks=4, num_heads=8),
            prediction_head=PredictionHeadConfig(task_type='regression'),
        )
        model = Octopus(llm_model=llm, config=config)
    """
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    prediction_head: PredictionHeadConfig = field(default_factory=PredictionHeadConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'OctopusConfig':
        """Create config from dictionary."""
        encoder_cfg = EncoderConfig(**config_dict.get('encoder', {}))
        patching_cfg = PatchingConfig(**config_dict.get('patching', {}))
        fusion_cfg = FusionConfig(**config_dict.get('fusion', {}))
        prediction_head_cfg = PredictionHeadConfig(**config_dict.get('prediction_head', {}))
        return cls(
            encoder=encoder_cfg,
            patching=patching_cfg,
            fusion=fusion_cfg,
            prediction_head=prediction_head_cfg
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'encoder': self.encoder.__dict__,
            'patching': self.patching.__dict__,
            'fusion': self.fusion.__dict__,
            'prediction_head': self.prediction_head.__dict__,
        }


# Preset configurations
@dataclass
class SmallConfig(OctopusConfig):
    """Small model configuration for testing/prototyping."""
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        hidden_dim=128, num_layers=3, dropout=0.1
    ))
    patching: PatchingConfig = field(default_factory=lambda: PatchingConfig(
        k_max=16, r_max=32, steps=2, gate_hidden_dim=128
    ))
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig(
        num_blocks=2, num_heads=4, dropout=0.1
    ))


@dataclass
class BaseConfig(OctopusConfig):
    """Base model configuration (default)."""
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        hidden_dim=256, num_layers=6, dropout=0.1
    ))
    patching: PatchingConfig = field(default_factory=lambda: PatchingConfig(
        k_max=32, r_max=64, steps=3, gate_hidden_dim=256
    ))
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig(
        num_blocks=4, num_heads=8, dropout=0.1
    ))


@dataclass
class LargeConfig(OctopusConfig):
    """Large model configuration."""
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        hidden_dim=512, num_layers=12, dropout=0.1
    ))
    patching: PatchingConfig = field(default_factory=lambda: PatchingConfig(
        k_max=64, r_max=128, steps=4, gate_hidden_dim=512
    ))
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig(
        num_blocks=6, num_heads=16, dropout=0.1
    ))

