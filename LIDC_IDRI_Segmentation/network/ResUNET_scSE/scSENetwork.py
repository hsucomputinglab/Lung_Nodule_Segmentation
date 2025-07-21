import torch
import torch.nn as nn
import torch.nn.functional as F


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


class UpSampleFusion(nn.Module):
    """
    Upsampling module using Multi-Scale Feature Fusion and bilinear interpolation.
    """

    def __init__(self, in_ch):
        super(UpSampleFusion, self).__init__()
        self.msf = MultiScaleFeatureFusion(in_ch)

    def forward(self, x):
        x = self.msf(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x


class ChannelSELayer(nn.Module):
    """
    Channel Squeeze and Excitation (SE) Layer.
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        reduced_channels = num_channels // reduction_ratio
        self.fc1 = nn.Linear(num_channels, reduced_channels, bias=True)
        self.fc2 = nn.Linear(reduced_channels, num_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, H, W = x.size()
        squeeze_tensor = x.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(x, fc_out_2.view(a, b, 1, 1))
        return output_tensor

        # excitation = self.fc1(squeeze_tensor)
        # excitation = self.relu(excitation)
        # excitation = self.fc2(excitation)
        # excitation = self.sigmoid(excitation).view(batch_size, num_channels, 1, 1)
        # output = x * excitation
        # return output


class SpatialSELayer(nn.Module):
    """
    Spatial Squeeze and Excitation (SE) Layer.
    """
    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor
    # def __init__(self, num_channels):
    #     super(SpatialSELayer, self).__init__()
    #     self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
    #     self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     squeeze_tensor = self.conv(x)
    #     excitation = self.sigmoid(squeeze_tensor)
    #     output = x * excitation
    #     return output


class ChannelSpatialSELayer(nn.Module):
    """
    Concurrent Spatial and Channel 'Squeeze & Excitation' Layer.
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, x):
        output = torch.max(self.cSE(x), self.sSE(x))
        return output
        #  cse_output = self.cSE(x)
        # sse_output = self.sSE(x)
        # output = torch.max(cse_output, sse_output)
        # return output


class scSENetwork(nn.Module):
    """
    U-Net architecture with residual blocks and concurrent spatial and channel SE layers.
    """
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 initial_filters: int = 32,
                 num_levels: int = 5):
        super().__init__()
        self.num_levels = num_levels
        self.initial_filters = initial_filters

        # -------- Encoder --------
        self.encoder = nn.ModuleList()
        self.poolings = nn.ModuleList()
        encoder_channels = []
        for level in range(num_levels):
            in_f = in_channels if level == 0 else initial_filters * (2 ** (level - 1))
            out_f = initial_filters * (2 ** level)
            encoder_channels.append(out_f)
            self.encoder.append(nn.Sequential(
                ResidualBlock(in_f, out_f),
                ChannelSpatialSELayer(out_f)
            ))
            if level < num_levels - 1:
                self.poolings.append(nn.MaxPool2d(kernel_size=2))

        # -------- Bottleneck --------
        self.bottleneck = nn.Sequential(
            ResidualBlock(encoder_channels[-1], encoder_channels[-1]),
            ChannelSpatialSELayer(encoder_channels[-1]),
            MultiScaleFeatureFusion(encoder_channels[-1]),  # MSFF with 4 dilations + concat + 1×1 → same channels
            nn.Dropout(0.1)
        )

        # -------- Decoder --------
        self.upsamples = nn.ModuleList()
        self.decoder   = nn.ModuleList()
        # rev_channels = list(reversed(encoder_channels))
        # dec_ch = encoder_channels[-1]
        # after bottleneck MSFF, channels = encoder_channels[-1]//2
        bottleneck_out_ch = encoder_channels[-1] // 2
        rev_channels = list(reversed(encoder_channels))
        dec_ch = bottleneck_out_ch

        for i in range(num_levels):
            if i == 0:
                # Up-step 0: MSFF → halve channels → bilinear ×2
                self.upsamples.append(nn.Sequential(
                    MultiScaleFeatureFusion(dec_ch),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                ))
                out_ch = dec_ch // 2

            # elif i == 1:
            else:
                # Up-step 1-4: MSFF → halve → upsample → conv 3×3 → quarter channels
                self.upsamples.append(nn.Sequential(
                    MultiScaleFeatureFusion(dec_ch),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dec_ch // 2, dec_ch // 4, kernel_size=3, padding=1),
                    nn.BatchNorm2d(dec_ch // 4),
                    nn.ReLU(inplace=True)
                ))
                out_ch = dec_ch // 4

            # else:
            #     # Up-step 2–4: MSFF bilinear ×2 → conv 3×3 → half channels
            #     self.upsamples.append(nn.Sequential(
            #         MultiScaleFeatureFusion(dec_ch),
            #         nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            #         nn.Conv2d(dec_ch, dec_ch // 2, kernel_size=3, padding=1),
            #         nn.BatchNorm2d(dec_ch // 2),
            #         nn.ReLU(inplace=True)
            #     ))
            #     out_ch = dec_ch // 2

            # After upsample, concatenate skip from encoder, then ResidualBlock
            self.decoder.append(ResidualBlock(out_ch + rev_channels[i], rev_channels[i]))
            dec_ch = rev_channels[i]

        # -------- Output head --------
        self.final_conv = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # print(f"Input: {x.shape}")
        encoder_feats = []
        out = x

        # Encoder path
        for i, enc in enumerate(self.encoder):
            out = enc(out)
            # print(f"Enc-{i}: {out.shape}")
            encoder_feats.append(out)
            if i < len(self.poolings):
                out = self.poolings[i](out)
                # print(f"Pool-{i}: {out.shape}")

        # Bottleneck
        out = self.bottleneck(out)
        # print(f"Bottleneck: {out.shape}")

        # Decoder path
        for i, up in enumerate(self.upsamples):
            out = up(out)
            # print(f"Up-step {i} pre-skip: {out.shape}")
            skip = encoder_feats[-(i + 1)]
            if out.shape[2:] != skip.shape[2:]:
                dy = skip.size(2) - out.size(2)
                dx = skip.size(3) - out.size(3)
                out = F.pad(out, [dx//2, dx-dx//2, dy//2, dy-dy//2])
                # print(f"Up-step {i} after pad: {out.shape}")
            out = torch.cat([out, skip], dim=1)
            # print(f"Up-step {i} concat: {out.shape}")
            out = self.decoder[i](out)
            # print(f"Dec-{i}: {out.shape}")

        # Final conv
        out = self.final_conv(out)
        # print(f"Final: {out.shape}")
        # return [out] #For Training
        return out #For Validation
    # def __init__(
    #     self, in_channels, out_channels, base_channels=[64, 128, 256, 512, 1024]
    # ):
    #     super(scSENetwork, self).__init__()

    #     # Encoder path
    #     self.conv1 = ResidualBlock(in_channels, base_channels[0])
    #     self.scse1 = ChannelSpatialSELayer(base_channels[0])
    #     self.pool1 = nn.MaxPool2d(2)
    #     self.conv2 = ResidualBlock(base_channels[0], base_channels[1])
    #     self.scse2 = ChannelSpatialSELayer(base_channels[1])
    #     self.pool2 = nn.MaxPool2d(2)
    #     self.conv3 = ResidualBlock(base_channels[1], base_channels[2])
    #     self.scse3 = ChannelSpatialSELayer(base_channels[2])
    #     self.pool3 = nn.MaxPool2d(2)
    #     self.conv4 = ResidualBlock(base_channels[2], base_channels[3])
    #     self.scse4 = ChannelSpatialSELayer(base_channels[3])
    #     self.pool4 = nn.MaxPool2d(2)
    #     self.conv5 = ResidualBlock(base_channels[3], base_channels[4])
    #     self.scse5 = ChannelSpatialSELayer(base_channels[4])

    #     # Decoder path
    #     self.up6 = UpSampleFusion(base_channels[4])
    #     self.conv6 = ResidualBlock(base_channels[4], base_channels[3])
    #     self.up7 = UpSampleFusion(base_channels[3])
    #     self.conv7 = ResidualBlock(base_channels[3], base_channels[2])
    #     self.up8 = UpSampleFusion(base_channels[2])
    #     self.conv8 = ResidualBlock(base_channels[2], base_channels[1])
    #     self.up9 = UpSampleFusion(base_channels[1])
    #     self.conv9 = ResidualBlock(base_channels[1], base_channels[0])
    #     self.conv10 = nn.Conv2d(base_channels[0], out_channels, kernel_size=1)

    # def forward(self, x):
    #     # Encoder
    #     c1 = self.conv1(x)
    #     s1 = self.scse1(c1)
    #     p1 = self.pool1(s1)

    #     c2 = self.conv2(p1)
    #     s2 = self.scse2(c2)
    #     p2 = self.pool2(s2)

    #     c3 = self.conv3(p2)
    #     s3 = self.scse3(c3)
    #     p3 = self.pool3(s3)

    #     c4 = self.conv4(p3)
    #     s4 = self.scse4(c4)
    #     p4 = self.pool4(s4)

    #     c5 = self.conv5(p4)
    #     s5 = self.scse5(c5)

    #     # Decoder
    #     up6 = self.up6(s5)
    #     merge6 = torch.cat([up6, s4], dim=1)
    #     c6 = self.conv6(merge6)

    #     up7 = self.up7(c6)
    #     merge7 = torch.cat([up7, s3], dim=1)
    #     c7 = self.conv7(merge7)

    #     up8 = self.up8(c7)
    #     merge8 = torch.cat([up8, s2], dim=1)
    #     c8 = self.conv8(merge8)

    #     up9 = self.up9(c8)
    #     merge9 = torch.cat([up9, s1], dim=1)
    #     c9 = self.conv9(merge9)

    #     output = self.conv10(c9)
    #     return output


# ------------------------------------------------------------------------
# 1) A lightweight CoarseStage: you can reuse your scSENetwork with 
#    fewer base_channels or smaller input.
# ------------------------------------------------------------------------
# class CoarseStage(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         # reuse your Residual+scSE U-Net but with half input size
#         # here we simply reuse scSENetwork but you can reduce base_channels
#         self.net = scSENetwork(in_ch, out_ch, base_channels=[32,64,128,256,512])

#     def forward(self, x):
#         # downsample the input by 2
#         x_ds = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
#         logits_ds = self.net(x_ds)      # e.g. [B, classes, 256,256]
#         return logits_ds


# ------------------------------------------------------------------------
# 2) A FineStage: concatenates the upsampled coarse mask with original image
# ------------------------------------------------------------------------
# class FineStage(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         # input channels = original_channels + num_classes (coarse mask)
#         self.net = scSENetwork(in_ch + out_ch, out_ch,
#                                base_channels=[64,128,256,512,1024])

#     def forward(self, x, coarse_logits):
#         # upsample coarse logits back to full res
#         coarse_up = F.interpolate(coarse_logits,
#                                   size=x.shape[-2:],
#                                   mode="bilinear",
#                                   align_corners=False)
#         # 2) Normalize to probabilities (IMPORTANT!)
#         #    - For binary segmentation use sigmoid
#         #    - For multi-class use softmax along channel dim
#         coarse_prob = torch.sigmoid(coarse_up)  
#         # If you had >1 class: 
#         # coarse_prob = F.softmax(coarse_up, dim=1)

#         # 3) Concatenate image + normalized coarse probability map
#         x_cat = torch.cat([x, coarse_prob], dim=1)

#         # 4) Pass through the fine U-Net
#         fine_logits = self.net(x_cat)
#         return fine_logits


# ------------------------------------------------------------------------
# 3) Full Cascade: runs coarse then fine
# ------------------------------------------------------------------------
# class CTFscSENet(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.coarse = CoarseStage(in_ch, out_ch)
#         self.fine   = FineStage(in_ch, out_ch)

#     def forward(self, x):
#         coarse_logits = self.coarse(x)
#         fine_logits   = self.fine(x, coarse_logits)
#         return coarse_logits, fine_logits