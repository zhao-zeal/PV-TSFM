"""BasicTS interface for the published Cross-Unet computation."""

from types import SimpleNamespace

from .official_model import Model


class CrossUnet(Model):
    def __init__(self, config):
        params = dict(config)
        self.power_only = params.pop('power_only')
        history_features = params.pop('history_features')
        weather_features = params.pop('weather_features')
        params['seq_len'] = params.pop('input_len')
        params['pred_len'] = params.pop('output_len')
        params['enc_in'] = history_features + (weather_features if params['useweather'] else 0)
        super().__init__(SimpleNamespace(**params))

    def forward(self, inputs, future_weather, history_weather, correlation_history):
        """Power must be the last input channel; all covariates are explicit.

        inputs / correlation_history: [B, input_len, history_features].
        future_weather / history_weather: [B, input_len, weather_features].
        correlation_history is the earlier history block chosen by the loader.
        No targets or decoder labels are supplied to the model.
        """
        prediction = super().forward(
            inputs, None, future_weather, None, history_weather, correlation_history)
        return prediction[..., -1:] if self.power_only else prediction
