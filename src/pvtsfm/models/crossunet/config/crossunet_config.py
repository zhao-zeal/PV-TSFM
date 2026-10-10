"""Effective settings after use_cross_unet.py's per-horizon overrides."""

from dataclasses import dataclass

from basicts.configs import BasicTSModelConfig


@dataclass
class CrossUnetConfig(BasicTSModelConfig):
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
