import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'Discriminator': [(3, 196, 1, 32), (196, 196, 2, 16), (196, 196, 1, 16), 
        (196, 196, 2, 8), (196, 196, 1, 8), (196, 196, 1, 8), (196, 196, 1, 8), (196, 196, 2, 4)],
    "Generator": ["deconv", 196, 196, 196, "deconv", 196, "deconv", 3],
}


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = [self._make_layers(setting) for setting in cfg["Discriminator"]]
        self.MaxPool = nn.MaxPool2d(4, padding=0, stride=4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x, extract_features=0):
        out = self.features[0](x)
        out = self.features[1](out)
        out = self.features[2](out)
        out = self.features[3](out)
        if extract_features == 4:
            h = F.max_pool2d(out, 4, 4)
            h = h.view(-1, 196*8*8)
            return h
        out = self.features[4](out)
        out = self.features[5](out)
        out = self.features[6](out)
        out = self.features[7](out)
        if extract_features == 8:
            h = F.max_pool2d(out, 4, 4)
            h = h.view(-1, 196*4*4)
            return h
        out = self.MaxPool(out)
        out = out.view(out.size(0), -1)
        out_fc1 = self.fc1(out)
        out_fc10 = self.fc10(out)
        return out_fc1, out_fc10

    def _make_layers(self, setting):
        layers = []
        in_chan, out_chan, stride, out_size = setting
        layers += [nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, stride=stride),
                    nn.LayerNorm((out_chan, out_size, out_size)),
                    nn.LeakyReLU(inplace=True)]
        return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.features = self._make_layers(cfg["Generator"])
        self.linear = nn.Linear(100, 196*4*4)

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(out.size(0), 196, 4, 4)
        out = self.features(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "deconv":
                layers += [nn.ConvTranspose2d(196, 196, kernel_size=4, padding=1, stride=2),
                           nn.BatchNorm2d(196),
                           nn.ReLU(inplace=True)]
            elif x != 3:
                layers += [nn.Conv2d(196, x, kernel_size=3, padding=1, stride=1),
                           nn.BatchNorm2d(196),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(196, x, kernel_size=3, padding=1, stride=1),
                            nn.Tanh()]
        return nn.Sequential(*layers)
