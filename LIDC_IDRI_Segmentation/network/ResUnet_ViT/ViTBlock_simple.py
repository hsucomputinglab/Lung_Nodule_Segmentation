import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, H'*W')
        x = x.transpose(1, 2)  # (B, H'*W', embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.n_heads, C // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self, dim, n_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBottleneck(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        img_size=32, #16 #64 - Image Size:512-Patch Size:32. 64 - 4
        patch_size=8, #4 - 256 ; 32 - 512 # divisible to be 16. 
        # N_patches = (img_size / patch_size) ^ 2
        embed_dim=128, #128 - Tips: embed_dim = sqrt(N_patches)
        depth=8,
        n_heads=16, #16
        mlp_ratio=4.0, #4.0 - This affect: hiddem_dim = embed_dim * mlp_ratio
        # Keep the hidden_dim around 128-512.
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape and permute for segmentation head
        B, N_patches, _ = x.shape
        H_patches = W_patches = int(N_patches ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, self.embed_dim, H_patches, W_patches)
        # x = x.permute(0, 2, 1).reshape(
        #     B, self.embed_dim, H // self.patch_size, W // self.patch_size
        # )
        x = self.seg_head(x)

        # Upsample to original image size
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return x

# class VisionTransformer(nn.Module):
#     def __init__(
#         self,
#         img_size=224,
#         patch_size=16,
#         in_channels=3,
#         n_classes=1000,
#         embed_dim=768,
#         depth=12,
#         n_heads=12,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#     ):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
#         )
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         self.blocks = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate
#                 )
#                 for _ in range(depth)
#             ]
#         )

#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, n_classes)

#     def forward(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for block in self.blocks:
#             x = block(x)

#         x = self.norm(x)
#         x = self.head(x[:, 0])
#         return x

# # Example usage
# model = VisionTransformerForSegmentation(
#     img_size=224, patch_size=16, in_ch=3, n_classes=21
# )
# input_tensor = torch.randn(1, 3, 224, 224)
# output = model(input_tensor)
# print(output.shape)  # Should be (1, 21, 224, 224)
