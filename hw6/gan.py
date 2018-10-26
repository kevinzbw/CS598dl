import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'Discriminator': [(3, 196, 1, 32), (196, 196, 2, 16), (196, 196, 1, 16), 
        (196, 196, 2, 8), (196, 196, 1, 8), (196, 196, 1, 8), (196, 196, 1, 8), (196, 196, 2, 4)],
    "Generator": ["deconv", 196, 196, 196, "deconv", 196, "deconv", 3],
}


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.features = [self._make_layers(setting) for setting in cfg["Discriminator"]]
#         self.MaxPool = nn.MaxPool2d(4, padding=0, stride=4)
#         self.fc1 = nn.Linear(196, 1)
#         self.fc10 = nn.Linear(196, 10)

#     def forward(self, x, extract_features=0):
#         out = self.features[0](x)
#         out = self.features[1](out)
#         out = self.features[2](out)
#         out = self.features[3](out)
#         if extract_features == 4:
#             h = F.max_pool2d(out, 4, 4)
#             h = h.view(-1, 196*8*8)
#             return h
#         out = self.features[4](out)
#         out = self.features[5](out)
#         out = self.features[6](out)
#         out = self.features[7](out)
#         if extract_features == 8:
#             h = F.max_pool2d(out, 4, 4)
#             h = h.view(-1, 196*4*4)
#             return h
#         out = self.MaxPool(out)
#         out = out.view(out.size(0), -1)
#         out_fc1 = self.fc1(out)
#         out_fc10 = self.fc10(out)
#         return out_fc1, out_fc10

#     def _make_layers(self, setting):
#         layers = []
#         in_chan, out_chan, stride, out_size = setting
#         layers += [nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, stride=stride),
#                     nn.LayerNorm((out_chan, out_size, out_size)),
#                     nn.LeakyReLU(inplace=True)]
#         return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        # conv4, conv8 = conv2
        # conv5, conv6, conv7 = conv3
        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.MaxPool = nn.MaxPool2d(4, padding=0, stride=4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)
        self.ln1 = nn.LayerNorm((196, 32, 32))
        self.ln2 = nn.LayerNorm((196, 16, 16))
        self.ln3 = nn.LayerNorm((196, 16, 16))
        self.ln4 = nn.LayerNorm((196, 8, 8))
        self.ln5 = nn.LayerNorm((196, 8, 8))
        self.ln6 = nn.LayerNorm((196, 8, 8))
        self.ln7 = nn.LayerNorm((196, 8, 8))
        self.ln8 = nn.LayerNorm((196, 4, 4))

    def forward(self, x, extract_features=0):
        # 1
        out = self.conv1(x)
        out = self.ln1(out)
        out = F.leaky_relu(out)
        # 2
        out = self.conv2(out)
        out = self.ln2(out)
        out = F.leaky_relu(out)
        # 3
        out = self.conv3(out)
        out = self.ln3(out)
        out = F.leaky_relu(out)
        # 4
        out = self.conv4(out)
        out = self.ln4(out)
        out = F.leaky_relu(out)
        if extract_features == 4:
            h = F.max_pool2d(out, 4, 4)
            h = h.view(out.size(0), -1)
            return h
        # 5
        out = self.conv5(out)
        out = self.ln5(out)
        out = F.leaky_relu(out)
        # 6
        out = self.conv6(out)
        out = self.ln6(out)
        out = F.leaky_relu(out)
        # 7
        out = self.conv7(out)
        out = self.ln7(out)
        out = F.leaky_relu(out)
        # 8
        out = self.conv8(out)
        out = self.ln8(out)
        out = F.leaky_relu(out)
        if extract_features == 8:
            h = F.max_pool2d(out, 4, 4)
            h = h.view(out.size(0), -1)
            return h

        out = self.MaxPool(out)

        out = out.view(out.size(0), -1)
        out_fc1 = self.fc1(out)
        out_fc10 = self.fc10(out)

        return out_fc1, out_fc10

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
