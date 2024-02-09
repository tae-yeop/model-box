"""
https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py
https://github.dev/huggingface/pytorch-image-models
"""
import torch
import torch.nn as nn

from typing import Optional, Tuple
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
class ViMEncoderOutput(ModelOutput):
    last_hidden_state = None
    hidden_states = None


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.bias = config.patch_bias
        self.img_size = config.img_size
            
        # 어차피 디폴트 bias = True
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, 
                                    kernel_size=self.patch_size, stride=self.patch_size, bias=self.bias)

        patch_size = (self.patch_size, self.patch_size)
        if self.img_size is not None:
            img_size = (self.img_size, self.img_size)
            self.grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            

    def forward(self, x):
        embeddings = self.projection(x).flatten(2).transpose(1,2)
        return embeddings

# positional embeddings
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        

        self
        # 간단하게 한다면
        # cls True이면 
        # self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, config.hidden_size))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)


class ViMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = 

class VisionMamba(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.patch_embedding = PatchEmbeddings()
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Positional Embedding
        pos_embed_cls = config.pos_embed_cls
        pos_embed_cls()
        
        
    def forward(self, x):
        embedding_output = self.patch_embedding(x)
        
        return ModelOutput(last,
                           hidden_states=hidden_states)

class VimPreTrainedModel(nn.Module):

    def _initialize_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
        elif isinstance(module, ViTEmbeddings):
            ...
        
    def init_weights(self):
        self.apply(self._initialize_weights)

class VimForImageClassification(VimPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vim = VisionMamba(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.init_weights()

    def forward(self, x):
        outputs = self.vim(x)
        outputs = self.classifier(outputs[0])
        return outputs