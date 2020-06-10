import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_size):
        super(Generator, self).__init__()

        self._network = nn.Sequential(
            # From input z into 1024 x 4 x4.
            nn.ConvTranspose2d(z_size,
                               1024,
                               4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # From 1024 x 4 x4 into 512 x 8 x 8.
            nn.ConvTranspose2d(1024,
                               512,
                               4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # From 512 x 8 x 8 into 256 x 16 x 16.
            nn.ConvTranspose2d(512,
                               256,
                               4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # From 256 x 16 x 16 into 128 x 32 x 32.
            nn.ConvTranspose2d(256,
                               128,
                               4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # From 128 x 32 x 32 into 3 x 64 x 64
            nn.ConvTranspose2d(128,
                               3,
                               4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self._network(input)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self._network = nn.Sequential(
            # From 3 x 64 x 64 into 128 x 32 x 32.
            nn.Conv2d(3, 128, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # From 128 x 32 x 32 into 256 x 16 x 16.
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # From 256 x 16 x 16 into 512 x 8 x 8.
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # From 512 x 8 x 8 into 1024 x 4 x 4.
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # From 1024 x 4 x 4 into.
            nn.Conv2d(1024, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self._network(input)


def init_weights(module):
    classname = module.__class__.__name__

    if 'Conv' in classname:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(module.bias.data, 0)
