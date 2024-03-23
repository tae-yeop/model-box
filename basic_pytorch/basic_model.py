"""
기초적인 모델 구현과 초기화 부분
"""

import torch
import torch.nn as nn


init_method_dict = {
    'zero': nn.init.zeros_,
    'constant': nn.init.constant_,
    'uniform': nn.init.uniform_,
    'kaiming-he': nn.init.kaiming_normal_,
    'xavier-glorot': nn.init.xavier_uniform_,
    
}

class Model(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        # inplace 하면 추가텐서를 메모리에 할당하지 않아서 메모리 절약 효과
        # https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948/10
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d()
        self.linear = nn.Linear(model_args.in_dim, model_args.out_dim, bias=model_args.bias)

        self.reset_parameters(model_args.init_method)
        
    def reset_parameters(self, init_method):
        """
        초기화는 모델별로 따로 하는 방법이 기본이다
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if model_args.bias is not None:
                    

    def forward(self,x):
        return self.linear_layer(x)