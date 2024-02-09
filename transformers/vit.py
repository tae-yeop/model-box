"""
https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
https://github.dev/huggingface/transformers
"""
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from dataclasses import dataclass
from collections import OrderedDict
class ModelOutput(OrderedDict):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.__dict__.keys())[key]
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        

@dataclass
class ViTEncoderOutput(ModelOutput):
    last_hidden_state = None
    hidden_states = None



class ViTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.mean_pool = config.mean_pool
        
    def forward(self, hidden_states):
        # first token or mean token
        hidden_states = hidden_states.mean(dim=1) if self.mean_pool else hidden_states[:, 0]
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = 
        self.

        self.layernorm_

        
class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.norm = nn.LayerNorm(self.hidden_size)
        self.layers = nn.ModuleList([ViTLayer(config) for _ in range(self.num_hidden_layers)])

    def forward(self, hidden_states, output_hidden_states: bool = False):
        all_hidden_states = () if output_hidden_states else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )


        return ViTEncoderOutput(last_hidden_state=hidden_states,
                                hidden_states=all_hidden_states)
            
class ViTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_height, image_width = config.image_size, config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.drop_rate = config.drop_rate

        
        assert image_height % self.patch_size == 0 and image_width % self.patch_size == 0, 'Image dimensions must be divisible by the patch size'
        num_patches = (image_height // self.patch_size) * (image_width // self.patch_size)
        patch_dim = self.num_channels * self.patch_size * self.patch_size

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.LayerNorm(patch_dim), # pre-norm
            nn.Linear(patch_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches +1, self.hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.dropout = nn.Dropout(self.drop_rate)

        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if config.add_pooling_layer else None

    def forwrad(self, pixel_values):
        embedding_output = self.patch_embedding(pixel_values)
        batch, seq_len,_ = embedding_output.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch)
        embedding_output = torch.cat((cls_tokens, embedding_output), dim=1)
        # x가 더 작은 크기가 오면 여기서 n
        embedding_output += self.pos_embedding[:, :(seq_len+1)]
        embedding_output = self.dropout(embedding_output)

        sequence_output = self.encoder(embedding_output)
        sequence_output = self.layernorm(sequence_output)

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return sequence_output, pooled_output
        
class ViTForImageClassification():
    def __init__(self, config):
        super().__init__()

        
    def forward():
        