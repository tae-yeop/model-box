from resnet import ResidualWrapper
import torch.nn as nn

class ConvMixerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size # 5
        
        self.model = nn.Sequential(ResidualWrapper(nn.Sequential(
                    nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, groups=self.hidden_size, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(self.hidden_size)
                )),
                nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_size))

    def forward(self, x):
        return self.model(x)

class ConvMixer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.n_classes = config.n_classes
        
        # PatchEmbedding
        self.projection = nn.Conv(self.num_channels, self.hidden_size, 
                                  kernel_size=self.patch_size, stride=self.patch_size)
        self.prenorm = nn.Sequential(nn.GELU(), nn.BatchNorm2d(self.hidden_size))
        self.encoder = nn.ModuleList([ConvMixerLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(),
                                    nn.Linear(self.hidden_size, self.n_classes))
    def forward(self, x):
        x = self.projection(x)
        out = self.prenomr(x)
        out = self.encoder(out)
        out = self.pooler(out)
        return out