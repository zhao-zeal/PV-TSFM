import torch
import numpy as np
from torch import nn
from transformers import CLIPProcessor, CLIPModel

class VLMManager:
    """Official CLIP branch; all resources loaded from project-local paths."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu}' if config.use_gpu else 'cpu')
        self._init_clip()
        self.model.to(self.device)

    @staticmethod
    def _set_requires_grad(model, value):
        for parameter in model.parameters():
            parameter.requires_grad = value

    def process_inputs(self, B, images, prompts):
        return self._process_clip_inputs(B, images, prompts)

    def _init_clip(self):
        CLIP_ARCH = self.config.vlm_path
        self.processor = CLIPProcessor.from_pretrained(CLIP_ARCH, local_files_only=True, use_fast=False)
        self.model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True, local_files_only=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 512
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77
        self.fused_feature_len = 9
        self._text_cache = {}
        # Enumerate the original NumPy rescale/normalization for all uint8 values.
        # Lookup on the GPU preserves its float32 rounding exactly.
        ramp = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, None, :], (3, 1, 256)).copy()
        lookup = self.processor.image_processor(
            images=ramp, do_resize=False, do_center_crop=False,
            input_data_format='channels_first', return_tensors='np')['pixel_values'][0]
        self._image_lookup = torch.from_numpy(lookup.reshape(3, 256)).to(self.device)
        self._image_channels = torch.arange(3, device=self.device)[None, :, None, None]
        self.multimodal_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ).to(self.device)


    def _process_clip_inputs(self, B, images, prompts):
        key = tuple(prompts)
        if key not in self._text_cache:
            self._text_cache[key] = self.processor.tokenizer(
                text=prompts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        # Keep official PIL bicubic resize/crop. Transfer the whole batch once,
        # then upload uint8 instead of float32 and normalize by exact lookup.
        pixels = self.processor.image_processor(
            images=images.detach().cpu().numpy(), do_rescale=False, do_normalize=False,
            input_data_format='channels_first', return_tensors='np')['pixel_values']
        pixels = torch.from_numpy(pixels).to(self.device)
        encoding = dict(self._text_cache[key])
        encoding['pixel_values'] = self._image_lookup[self._image_channels, pixels.long()]
        outputs = self.model(**encoding, output_hidden_states=True)
        text_features = outputs.text_embeds  # Shape: [B, hidden_size]
        image_features = outputs.image_embeds  # Shape: [B, hidden_size]
        return image_features, text_features  # Both shape: [B, hidden_size]
