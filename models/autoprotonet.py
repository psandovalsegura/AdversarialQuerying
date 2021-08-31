import math
import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True, activation='ReLU'):
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if retain_activation:
            if activation == 'ReLU':
                self.block.add_module("ReLU", nn.ReLU(inplace=True))
            elif activation == 'LeakyReLU':
                self.block.add_module("LeakyReLU", nn.LeakyReLU(0.1))
            elif activation == 'Softplus':
                self.block.add_module("Softplus", nn.Softplus())
        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
    def forward(self, x):
        out = self.block(x)
        return out

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_padding=0, retain_activation=True, activation='ReLU'):
        super(ConvTransposeBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        
        if retain_activation:
            if activation == 'ReLU':
                self.block.add_module("ReLU", nn.ReLU(inplace=True))
            elif activation == 'LeakyReLU':
                self.block.add_module("LeakyReLU", nn.LeakyReLU(0.1))
            elif activation == 'Softplus':
                self.block.add_module("Softplus", nn.Softplus())
        
    def forward(self, x):
        out = self.block(x)
        return out

class AutoProtoNetEmbedding(nn.Module):
    """ Model consists of an encoder and decoder
    """
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True, activation='ReLU', is_miniimagenet=False):
        super(AutoProtoNetEmbedding, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(x_dim, h_dim, activation=activation),
            ConvBlock(h_dim, h_dim, activation=activation),
            ConvBlock(h_dim, h_dim, activation=activation),
            ConvBlock(h_dim, z_dim, retain_activation=retain_last_activation, activation=activation),
        )

        self.decoder = nn.Sequential(
            ConvTransposeBlock(z_dim, h_dim),
            ConvTransposeBlock(h_dim, h_dim, output_padding=(0 if not is_miniimagenet else 1)),
            ConvTransposeBlock(h_dim, h_dim),
            ConvTransposeBlock(h_dim, x_dim)
        )

        self.embedding_shape = None

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        """ Forward on the AutoProtoNet produces an embedding
        """
        x = self.encoder(x)
        self.embedding_shape = [-1] + list(x.shape[-3:])
        return x.view(x.size(0), -1)

    def forward_decoder(self, e):
        x = self.decoder(e)
        return x

    def forward_plus_decoder(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x