"""Architecture settings from FusionSF's merged experiment configuration."""

from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class FusionSFConfig(BasicTSModelConfig):
    input_len: int = 24
    output_len: int = 24
    image_size: list = field(default_factory=lambda: [64, 64])
    patch_size: list = field(default_factory=lambda: [8, 8])
    frequencies: list = field(default_factory=lambda: [12, 31, 24])
    dim: int = 64
    depth: int = 12
    heads: int = 8
    mlp_ratio: int = 4
    ctx_channels: int = 1
    ts_channels: int = 1
    guide_channels: int = 15
    out_dim: int = 1
    dim_head: int = 64
    dropout: float = 0.4
    freq_type: str = 'lucidrains'
    pe_type: str = 'rope'
    num_mlp_heads: int = 1
    use_glu: bool = True
    use_self_attention: bool = True
    max_freq: int = 128
    ctx_masking_ratio: float = 0.85
    ts_masking_ratio: float = 0
    decoder_dim: int = 128
    decoder_depth: int = 4
    decoder_heads: int = 6
    decoder_dim_head: int = 128
    vq_in_ts: bool = False
    vq_in_ctx: bool = False
    vq_in_guide: bool = False
