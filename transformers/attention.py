"""

"""

import torch.nn as nn

class AttnProcessor(nn.Module):
    

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.heads = config.heads
        self.causal = config.causal
        self.query_dim = config.query_dim
        self.inner_dim = config.inner_dim
        self.out_dim = config.out_dim
        self.out_bias = config.out_bias
        
        linear_cls = config.linear_cls # [nn.Linear, LoRACompatibleLinear]
        self.to_q = linear_cls(self.query_dim, self.inner_dim, bias=config.attention_bias)


        self.to_out = nn.ModuleList([])
        self.to_out.append(linear_cls(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))
        
        # self.to_out = nn.Sequential(nn.Linear(out_dim, dim * 2, bias = False), nn.GLU()) if on_attn else nn.Linear(out_dim, dim, bias = False)

        self.processor = AttnProcessor()


    def forward(self, x):
        