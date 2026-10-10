"""Architecture settings from FusionSF's merged experiment configuration."""

from dataclasses import dataclass, field




@dataclass
class FusionSFConfig:
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

"""Effective settings after use_cross_unet.py's per-horizon overrides."""

from dataclasses import dataclass




@dataclass
class CrossUnetConfig:
    input_len: int = 96
    output_len: int = 96
    history_features: int = 1
    weather_features: int = 1
    seg_len: int = 24
    d_model: int = 256
    d_ff: int = 512
    e_layers: int = 3
    n_heads: int = 4
    factor: int = 10
    dropout: float = 0.05
    useweather: bool = True
    usenonlinearproject: bool = True
    usebottle: bool = True
    convmerge: bool = False
    swichchannel: bool = False
    twofilter: bool = True
    task_name: str = 'long_term_forecast'
    power_only: bool = True

    @classmethod
    def from_official_script(cls, station_name='KDASC', pred_len=96,
                             use_satell=False, deployment=False):
        """Resolve one official loop iteration, measured in 15-minute steps."""
        if station_name in ('KDASC', 'yulara'):
            history_features, weather_features = 1, 1
        else:
            history_features = 7
            weather_features = 1 if use_satell or deployment else 6
        seg_len = 12 if pred_len < 49 else (24 if pred_len <= 384 else 48)
        return cls(input_len=max(96, pred_len), output_len=pred_len,
                   history_features=history_features, weather_features=weather_features,
                   seg_len=seg_len)

"""Single-target model settings from the official ETTh1 96→96 MS script."""

from dataclasses import dataclass




@dataclass
class TimeXerPVConfig:
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
