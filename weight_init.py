import torch.nn as nn


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)