import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, attn_drp, drp):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(p=attn_drp)
        self.proj = nn.Linear(dim, dim)
        self.block_drop = nn.Dropout(p=drp)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim)
        )
        self.block_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.block_norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):

        shape = x.shape[: -1] + (3, self.n_heads, self.head_dim)
        qkv = self.qkv(self.block_norm1(x)).view(shape)  # batch_size, n_patches +label, img_npy, n_heads, heads_dim
        qkv = qkv.permute(2, 0, 3, 1, 4)  # img_npy, batch_size, n_heads, n_patches +label, heads_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (batch_size, n_heads, n_patches +label, heads_dim)
        attn = q @ k.transpose(-2, -1) * self.scale  # (batch_size, n_heads, n_patches +label, n_patches +label)
        attn = self.attn_drop(attn.softmax(dim=-1))  # (batch_size, n_heads, n_patches +label, n_patches +label)
        attn_out = attn @ v  # (batch_size, n_heads, n_patches +label, head_dim)
        attn_out = attn_out.transpose(1, 2)  # (batch_size, n_patches + label, n_heads, head_dim)
        attn_out = attn_out.flatten(2)  # (batch_size, n_patches + label, dim)
        # для визуализации
        #self.attn = attn

        att = self.block_drop(self.proj(attn_out))  # (batch_size, n_patches + label, dim)
        x = x + att
        mlp = self.block_drop(self.mlp(self.block_norm2(x)))  # (batch_size, n_patches + label, hidden_features)
        x = x + mlp

        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=1,
                 n_chan=1024,
                 n_classes=1,
                 dim=768,
                 depth=6,
                 n_heads=12,
                 mlp_ratio=4,
                 pos_1d=True,
                 hybr=False,
                 attn_drp=0.1,
                 drp=0.1
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_1d = pos_1d
        self.hybr = hybr
        if self.pos_1d:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, dim))
        else:
            n = img_size // patch_size
            self.pos_embed_w = nn.Parameter(torch.zeros(1, dim, 1, n))
            self.pos_embed_h = nn.Parameter(torch.zeros(1, dim, n + 1, 1))
        if not self.hybr:
            self.emb_proj = nn.Conv2d(n_chan,
                                  dim,
                                  kernel_size=patch_size,
                                  stride=patch_size
                                  )
        self.blocks = nn.ModuleList([Block(dim=dim,
                                           n_heads=n_heads,
                                           mlp_ratio=mlp_ratio,
                                           attn_drp=attn_drp,
                                           drp=drp
                                           )
                                     for _ in range(depth)
                                     ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.output = nn.Linear(dim, n_classes)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.output.bias, 0)
        if self.pos_1d:
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            nn.init.normal_(self.pos_embed_w, std=0.02)
            nn.init.normal_(self.pos_embed_h, std=0.02)
        nn.init.constant_(self.cls_token, 0)

    def forward(self, x):
        # получение размера батча
        batch_size = x.shape[0]
        if not self.hybr:
            x = self.emb_proj(x)  # (batch_size, dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, dim)

        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, label, dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, label + n_patches, dim)
        if self.pos_1d:
            x = x + self.pos_embed  # (batch_size, label + n_patches, dim)
        else:
            pos_embed_w = self.pos_embed_w.expand(-1, -1, self.img_size // self.patch_size + 1, -1)
            pos_embed_h = self.pos_embed_h.expand(-1, -1, -1, self.img_size // self.patch_size)
            pos = (pos_embed_w + pos_embed_h).flatten(2)[:, :, - (1 + self.n_patches):]
            x = x + pos.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # (batch_size, label + n_patches, dim)
        x = x[:,1:]
        #cls_first = x[:, 0]  # (batch_size, dim)
        # x = self.output(cls_first)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channel):
        depth = in_channel
        super(ASPP, self).__init__()
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=2, dilation=2)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4)
        self.conv_1x1_output = nn.Conv2d(depth * 4, depth*2, 1, 1)
        self.batchnorm = nn.BatchNorm2d(in_channel*2)
        self.action = nn.ReLU(inplace=True)
    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        cat = torch.cat([ atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        net = self.batchnorm(net)
        net = self.action(net)
        #net = self.cam(net) * net
        return net

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.resconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch)
        )
        self.action = nn.ReLU(inplace=True)
    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.resconv(input)
        x3 = x1 + x2
        x3 = self.action(x3)
        return x3

class TransConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(TransConv, self).__init__()
        self.aspp = ASPP(in_ch)
        self.up = nn.PixelShuffle(2)
    def forward(self,input):
        x1 = self.aspp(input)
        x2 = self.up(x1)
        return x2

class SMRUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SMRUNet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = DoubleConv(512, 1024)
        #self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up6 = TransConv(1024, 512)
        self.conv6 = DoubleConv(1024, 512)
        #self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = TransConv(512, 256)
        self.conv7 = DoubleConv(512, 256)
        # self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = TransConv(256, 128)
        self.conv8 = DoubleConv(256, 128)
        #self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = TransConv(128, 64)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.dropout = nn.Dropout2d(p=0.5)
        # self.vit = VIT.VisionTransformer(depth=12, n_heads=12, img_size=256, dim=768, patch_size=16, pos_1d=True,
        #                                   hybr=True, n_classes=7, n_chan=1024)

        self.vit =VisionTransformer() #depth=6, n_heads=12, img_size=8, dim=768, patch_size=1, pos_1d=True,
                                           #hybr=False, n_classes=1, n_chan=1024)
        self.conv_1024_768 = nn.Conv2d(1024, 768, kernel_size=1, stride=1)
       # self.conv_e_768_1024 = DoubleConv(768, 1024)
        self.conv_e_768_1024 = nn.Conv2d(768, 1024,kernel_size=1,stride=1)
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c1 = self.conv1(x)  #(batch,64,256,256)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1) # (batch,128,128,128)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)  # (batch,256,64,64)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)  # (batch,512,32,32)
        mid1 = self.dropout(c4)
        p4 = self.pool4(mid1)
        c5 = self.conv5(p4)  #(batch,1024,16,16)
       # mid_16_768 = self.conv_1024_768(c5)
        mid_vit = self.vit(c5)
        B, n_patch, hidden = mid_vit.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        mid_vit = mid_vit.permute(0, 2, 1)
        mid_16_c768 = mid_vit.contiguous().view(B, hidden, h, w)  # (batch,768,16,16)
        mid_16_c1024 = self.conv_e_768_1024(mid_16_c768)
        mid2 = self.dropout(mid_16_c1024)
        up_6 = self.up6(mid2)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10

#==================================================================================
# if __name__ == '__main__':
#     model = Unet(1, 1)
#     x = torch.randn([8,1,128,128])
#     out = model(x)
#     print(out.shape)