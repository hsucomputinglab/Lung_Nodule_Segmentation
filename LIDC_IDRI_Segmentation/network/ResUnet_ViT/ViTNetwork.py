import torch
import torch.nn as nn
import torch.nn.functional as F

from .ViTBlock_simple import ViTBottleneck


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-Scale Feature Fusion module using atrous (dilated) convolutions.
    """

    def __init__(self, in_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        depth = in_channels

        self.atrous_block1 = nn.Conv2d(
            in_channels, depth, kernel_size=3, padding=1, dilation=1
        )
        self.atrous_block2 = nn.Conv2d(
            in_channels, depth, kernel_size=3, padding=2, dilation=2
        )
        self.atrous_block3 = nn.Conv2d(
            in_channels, depth, kernel_size=3, padding=3, dilation=3
        )
        self.atrous_block4 = nn.Conv2d(
            in_channels, depth, kernel_size=3, padding=4, dilation=4
        )
        self.conv_1x1_output = nn.Conv2d(depth * 4, depth // 2, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(depth // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)
        concat = torch.cat(
            [atrous_block1, atrous_block2, atrous_block3, atrous_block4], dim=1
        )
        output = self.conv_1x1_output(concat)
        output = self.batchnorm(output)
        output = self.relu(output)
        return output


class ResidualBlock(nn.Module):
    """
    Residual Block with two convolutional layers and a skip connection.
    """

    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()

        self.conv_sequence = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        self.skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_out = self.conv_sequence(x)
        skip_out = self.skip_connection(x)
        output = self.relu(conv_out + skip_out)
        return output


class TransConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TransConv, self).__init__()
        self.msf = MultiScaleFeatureFusion(in_ch)

    def forward(self, x):
        x = self.msf(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x


class ChannelSELayer(nn.Module):
    """
    Channel Squeeze and Excitation (SE) Layer.
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        reduced_channels = num_channels // reduction_ratio
        self.fc1 = nn.Linear(num_channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, H, W = x.size()
        squeeze_tensor = x.view(batch_size, num_channels, -1).mean(dim=2)
        excitation = self.fc1(squeeze_tensor)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch_size, num_channels, 1, 1)
        output = x * excitation
        return output


class SpatialSELayer(nn.Module):
    """
    Spatial Squeeze and Excitation (SE) Layer.
    """

    def __init__(self, num_channels):
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        squeeze_tensor = self.conv(x)
        excitation = self.sigmoid(squeeze_tensor)
        output = x * excitation
        return output


class ChannelSpatialSELayer(nn.Module):
    """
    Concurrent Spatial and Channel 'Squeeze & Excitation' Layer.
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, x):
        cse_output = self.cSE(x)
        sse_output = self.sSE(x)
        output = torch.max(cse_output, sse_output)
        return output


class ViTNetwork(nn.Module):

    def __init__(self, in_ch, out_ch, base_channels=[64, 128, 256, 512, 1024]):
    # def __init__(self, in_ch, out_ch, base_channels=[32, 64, 128, 256, 512]):

        super().__init__()

        self.conv1 = ResidualBlock(in_ch, base_channels[0])
        self.scse1 = ChannelSpatialSELayer(base_channels[0])
        self.pool1 = nn.MaxPool2d(2, stride=2) #256
        self.conv2 = ResidualBlock(base_channels[0], base_channels[1])
        self.scse2 = ChannelSpatialSELayer(base_channels[1])
        self.pool2 = nn.MaxPool2d(2, stride=2) #128
        self.conv3 = ResidualBlock(base_channels[1], base_channels[2])
        self.scse3 = ChannelSpatialSELayer(base_channels[2])
        self.pool3 = nn.MaxPool2d(2, stride=2) #64
        self.conv4 = ResidualBlock(base_channels[2], base_channels[3])
        self.scse4 = ChannelSpatialSELayer(base_channels[3])
        self.pool4 = nn.MaxPool2d(2, stride=2) #32
        self.conv5 = ResidualBlock(base_channels[3], base_channels[4])

        self.bn = ViTBottleneck(base_channels[4], base_channels[4])

        self.up6 = TransConv(base_channels[4], base_channels[3])
        self.conv6 = ResidualBlock(base_channels[4], base_channels[3])
        # self.scse6 = scSEBlock(base_channels[3])
        self.up7 = TransConv(base_channels[3], base_channels[2])
        self.conv7 = ResidualBlock(base_channels[3], base_channels[2])
        # self.scse7 = scSEBlock(base_channels[2])
        self.up8 = TransConv(base_channels[2], base_channels[1])
        self.conv8 = ResidualBlock(base_channels[2], base_channels[1])
        # self.scse8 = scSEBlock(base_channels[1])
        self.up9 = TransConv(base_channels[1], base_channels[0])
        self.conv9 = ResidualBlock(base_channels[1], base_channels[0])
        # self.scse9 = scSEBlock(base_channels[0])
        self.conv10 = nn.Conv2d(base_channels[0], out_ch, 1)

    # def weight_init(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)

    def forward(self, input):  # INPUT: 1
        # print(f"Input Size: {input.size()}")  # ([8, 1, 128, 128])

        c1 = self.conv1(input)  # (batch,64,256,256)
        s1 = self.scse1(c1)
        # print(f"scSE 1: {s1.size()}")

        p1 = self.pool1(s1)

        c2 = self.conv2(p1)  # (batch,128,128,128)
        s2 = self.scse2(c2)
        # print(f"scSE 2: {s2.size()}")

        p2 = self.pool2(s2)

        c3 = self.conv3(p2)  # (batch,256,64,64)
        s3 = self.scse3(c3)
        # print(f"scSE 3: {s3.size()}")

        p3 = self.pool3(s3)

        c4 = self.conv4(p3)  # (batch,512,32,32)
        s4 = self.scse4(c4)
        # print(f"scSE 4: {s4.size()}")

        p4 = self.pool4(s4)
        # print(f"pool 4: {p4.size()}")

        c5 = self.conv5(p4)  # (batch,1024,16,16)
        # print(f"conv 5: {c5.size()}")
        s5 = self.bn(c5)
        # print(f"scSE 5: {s5.size()}")

        up_6 = self.up6(s5)
        # print(f"up6: {up_6.size()}")
        # print(f"s4: {s4.size()}")
        merge6 = torch.cat([up_6, s4], dim=1)
        c6 = self.conv6(merge6)
        # print(f"c6: {c6.size()}")
        # s6 = self.scse6(c6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, s3], dim=1)
        c7 = self.conv7(merge7)
        # print(f"c7: {c7.size()}")
        # s7 = self.scse7(c7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, s2], dim=1)
        c8 = self.conv8(merge8)
        # s8 = self.scse8(c8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, s1], dim=1)
        c9 = self.conv9(merge9)
        # s9 = self.scse9(c9)

        c10 = self.conv10(c9)
        return c10
