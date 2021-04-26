import torch
from torch import nn, cat
import torchvision
import math
from .basenet import MobileNetV2 , InvertedResidual

# Mobile Net encoder as Backbone and upsampling decoder
class UnetMobilenetV2(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True,
                 Dropout=.2, path='./data/mobilenet_v2.pth.tar'):
        super(UnetMobilenetV2, self).__init__()
        
        self.encoder = MobileNetV2(n_class=1000)
        
        self.num_classes = num_classes

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Conv2d(16, 3, 1)

        self.conv_score = nn.Conv2d(3, 1, 1)

        #only for compatibility
        self.dconv_final = nn.ConvTranspose2d(1, 1, 4, padding=1, stride=2)

        if pretrained:
            state_dict = torch.load(path)
            self.encoder.load_state_dict(state_dict)
        else: self._init_weights()

    def forward(self, x):
        for n in range(0, 2):
            x = self.encoder.features[n](x)
        x1 = x

        for n in range(2, 4):
            x = self.encoder.features[n](x)
        x2 = x

        for n in range(4, 7):
            x = self.encoder.features[n](x)
        x3 = x

        for n in range(7, 14):
            x = self.encoder.features[n](x)
        x4 = x

        for n in range(14, 19):
            x = self.encoder.features[n](x)
        x5 = x
        
        up1 = torch.cat([
            x4,
            self.dconv1(x)
        ], dim=1)
        up1 = self.invres1(up1)

        up2 = torch.cat([
            x3,
            self.dconv2(up1)
        ], dim=1)
        up2 = self.invres2(up2)

        up3 = torch.cat([
            x2,
            self.dconv3(up2)
        ], dim=1)
        up3 = self.invres3(up3)

        up4 = torch.cat([
            x1,
            self.dconv4(up3)
        ], dim=1)
        up4 = self.invres4(up4)
        x = self.conv_last(up4)
        x = self.conv_score(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()