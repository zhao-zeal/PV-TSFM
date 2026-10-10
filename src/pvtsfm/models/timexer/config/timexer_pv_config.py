"""Single-target model settings from the official ETTh1 96→96 MS script."""

from dataclasses import dataclass

from basicts.configs import BasicTSModelConfig


@dataclass
class TimeXerPVConfig(BasicTSModelConfig):
    input_len: int = 96
    output_len: int = 96
    nwp_channels: int = 15
    patch_len: int = 16
    d_model: int = 512
    d_ff: int = 512
    n_heads: int = 8
    e_layers: int = 2
    factor: int = 3
    dropout: float = 0.1
    activation: str = 'gelu'
    use_norm: bool = True
