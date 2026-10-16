"""STAEformer architecture adapted from the authors' published implementation."""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError('model_dim must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch = query.shape[0]
        query = torch.cat(self.query(query).split(self.head_dim, dim=-1), dim=0)
        key = torch.cat(self.key(key).split(self.head_dim, dim=-1), dim=0)
        value = torch.cat(self.value(value).split(self.head_dim, dim=-1), dim=0)
        scores = query @ key.transpose(-1, -2) / self.head_dim ** 0.5
        attended = torch.softmax(scores, dim=-1) @ value
        attended = torch.cat(attended.split(batch, dim=0), dim=-1)
        return self.output(attended)


class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, inputs, dim):
        inputs = inputs.transpose(dim, -2)
        attended = self.attention(inputs, inputs, inputs)
        hidden = self.attention_norm(inputs + self.attention_dropout(attended))
        output = self.feed_forward_norm(
            hidden + self.feed_forward_dropout(self.feed_forward(hidden)))
        return output.transpose(dim, -2)


class STAEformer(nn.Module):
    def __init__(self, num_nodes, in_steps, out_steps, steps_per_day=24,
                 input_dim=1, output_dim=1, input_embedding_dim=24,
                 tod_embedding_dim=24, dow_embedding_dim=24,
                 spatial_embedding_dim=0, adaptive_embedding_dim=80,
                 feed_forward_dim=256, num_heads=4, num_layers=3,
                 dropout=0.1, use_mixed_proj=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.use_mixed_proj = use_mixed_proj
        self.model_dim = (input_embedding_dim + tod_embedding_dim + dow_embedding_dim
                          + spatial_embedding_dim + adaptive_embedding_dim)

        self.input_projection = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim:
            self.node_embedding = nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_embedding)
        if adaptive_embedding_dim:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)

        if use_mixed_proj:
            self.output_projection = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_projection = nn.Linear(in_steps, out_steps)
            self.output_projection = nn.Linear(self.model_dim, output_dim)
        self.temporal_layers = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, inputs):
        if inputs.shape[1:3] != (self.in_steps, self.num_nodes):
            raise ValueError(
                f'Expected [batch,{self.in_steps},{self.num_nodes},channels], got {inputs.shape}')
        time_of_day = inputs[..., 1] if self.tod_embedding_dim else None
        day_of_week = inputs[..., 2] if self.dow_embedding_dim else None
        features = [self.input_projection(inputs[..., :self.input_dim])]
        if self.tod_embedding_dim:
            indices = (time_of_day * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            features.append(self.tod_embedding(indices))
        if self.dow_embedding_dim:
            features.append(self.dow_embedding(day_of_week.long().clamp(0, 6)))
        if self.spatial_embedding_dim:
            features.append(self.node_embedding.expand(inputs.shape[0], self.in_steps, -1, -1))
        if self.adaptive_embedding_dim:
            features.append(self.adaptive_embedding.expand(inputs.shape[0], -1, -1, -1))

        hidden = torch.cat(features, dim=-1)
        for layer in self.temporal_layers:
            hidden = layer(hidden, dim=1)
        for layer in self.spatial_layers:
            hidden = layer(hidden, dim=2)
        if self.use_mixed_proj:
            output = hidden.transpose(1, 2).reshape(
                inputs.shape[0], self.num_nodes, self.in_steps * self.model_dim)
            output = self.output_projection(output).view(
                inputs.shape[0], self.num_nodes, self.out_steps, -1)
            return output.transpose(1, 2)
        output = self.temporal_projection(hidden.transpose(1, 3))
        return self.output_projection(output.transpose(1, 3))
