import torch
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
        self.multimodal_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ).to(self.device)


    def _process_clip_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        text_features = outputs.text_embeds  # Shape: [B, hidden_size]
        image_features = outputs.image_embeds  # Shape: [B, hidden_size]
        return image_features, text_features  # Both shape: [B, hidden_size]
