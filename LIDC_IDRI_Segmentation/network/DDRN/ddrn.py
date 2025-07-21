import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels // 4, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(
            out_channels // 4, out_channels // 4, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(
            out_channels // 4, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ConvDeconvNetwork(nn.Module):
    def __init__(self):
        super(ConvDeconvNetwork, self).__init__()

        # Resnet-50 based encoder (convolution network)
        self.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = self._make_layer(64, 128, 3)
        self.conv2 = self._make_layer(128, 256, 3)
        self.conv3 = self._make_layer(256, 512, 3)
        self.conv4 = self._make_layer(512, 1024, 3)

        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Decoder (deconvolution network) with unpooling and residual blocks
        self.deconv4 = self._make_layer(1024, 512, 3)
        self.deconv3 = self._make_layer(512, 256, 3)
        self.deconv2 = self._make_layer(256, 128, 3)
        self.deconv1 = self._make_layer(128, 64, 3)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        # print(f"Input Size: {x.size()}") # B,1,512,512
        c1 = self.conv0(x)
        # print(f"C1 Size: {c1.size()}") # B,64,512,512
        c1 = self.bn1(c1)
        # print(f"C1 Size: {c1.size()}")  # B,64,512,512
        c1 = self.relu(c1)
        # print(f"C1 Size: {c1.size()}")  # B,64,512,512
        c1, indices1 = self.pool(c1)  
        # print(f"C1 Size: {c1.size()}")  # B,64,256,256
        # print(f"Indices1 Size: {indices1.size()}")  # B,64,256,256

        c2 = self.conv1(c1)
        # print(f"C2 Size: {c2.size()}")  # B,128,256,256
        c2, indices2 = self.pool(c2)  
        # print(f"C2 Size: {c2.size()}")  # B,128,128,128
        # print(f"Indices2 Size: {indices2.size()}")  # B,128,128,128

        c3 = self.conv2(c2)
        # print(f"C3 Size: {c3.size()}")  # B,256,128,128
        c3, indices3 = self.pool(c3)  
        # print(f"C3 Size: {c3.size()}")  # B,256,64,64
        # print(f"Indices3 Size: {indices3.size()}")  # B,256,64,64

        c4 = self.conv3(c3)
        # print(f"C4 Size: {c4.size()}")  # B,512,64,64
        c4, indices4 = self.pool(c4)  
        # print(f"C4 Size: {c4.size()}")  # B,512,32,32
        # print(f"Indices4 Size: {indices4.size()}")  # B,512,32,32

        c5 = self.conv4(c4)
        # print(f"C5 Size: {c5.size()}")  # B,1024,32,32

        d5 =self.deconv4(c5)
        # print(f"D5 Size: {d5.size()}")  # B,512,32,32

        d4 = self.unpool(d5, indices4)
        # print(f"D4 Size: {d4.size()}")  # B,512,64,64
        d4 = self.deconv3(d4)
        # print(f"D4 Size: {d4.size()}")  # B,256,64,64
        d3 = d4 + c3
        d3 = self.unpool(d4, indices3)
        # print(f"D3 Size: {d3.size()}")  # B,256,128,128
        d3 = self.deconv2(d3)
        # print(f"D3 Size: {d3.size()}")  # B,128,128,128
        d2 = d3 + c2
        d2 = self.unpool(d3, indices2)
        # print(f"D2 Size: {d2.size()}")  # B,128,256,256
        d2 = self.deconv1(d2)
        # print(f"D2 Size: {d2.size()}")  # B,64,64,64
        d1 = d2 + c1
        d1 = self.unpool(d2, indices1)
        # print(f"D1 Size: {d1.size()}")  # B,64,512,512

        out = self.final_conv(d1)  # B,1,512x512

        return out
