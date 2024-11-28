"""
Segformer的Encoder结构
"""

import math
from functools import partial
import torch
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob       = 1 - drop_prob                         # 分支的保留概率
    shape           = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets   # 生成分支筛选向量的形状
    random_tensor   = x.new_empty(shape).bernoulli_(keep_prob)  # 生成形状为shape的张量，并且使用0或者是1初始化张量，1出现的概率是keep_prob,1也就是保留path
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)                   # 这一步还是和另一个版本的drop_path一样，让输出的期望保持不变
    return x * random_tensor                            # 这一步就实际进行了剪枝操作

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        # 采样卷积层，OverlapPatchEmbed和ViT中的PatchEmbed的区别就是多了一个padding，这样能够让采样的图像拥有重叠区域
        # 并且重叠部分大小为patch_size // 2
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化OverlapPatchEmbed权重
        :param m:
        :return:
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2. / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):       # x (b, c, h, w)
        x = self.proj(x)        # x (b, embed, h/4, w/4) default
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # x (b, h*w/16, embed)
        x = self.norm(x)                        # x (b, h*w/16, embed) (b, n, c)

        return x, H, W

class EfficientSelfAttention(nn.Module):  # 感觉基本借鉴了PVT的注意力实现
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Wq
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        # 这里是Segformer相对ViT的创新，使用一个卷积层对原来的k缩放，减少了计算量
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        # Wk, Wv
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化权重
        :param m:
        :return:
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2. / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        # q (b, n, c) -> (b, n, heads, c/heads) -> (b, heads, n, c/heads) 产生多个头
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:  # reduction 维度收缩
            # x_ (b, c, n) -> (b, c, h, w)
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # x_ (b, c, h/r, w/r) -> (b, c, h*w/R) -> (b, h*w/R, c)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  #
            x_ = self.norm(x_)
            # kv (b, n/R, c*2) -> (b, n/R, 2, heads, c/heads) -> (2, b, heads, n/R, c/heads)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # kv (b, h*w, c*2) -> (b, n, 2, heads, c/heads) -> (2, b, heads, n, c/heads)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v (b, heads, n/R, c/heads)
        k, v = kv[0], kv[1]

        # attn (b, heads, n, c/heads) * (b, heads, c/heads, n/R) -> (b, heads, n, n/R)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x (b, heads, n, n/R) * (b, heads, n/R, c/heads) -> (b, heads, n, c/heads)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x (b, n, heads, c/heads) -> (b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)

        # x (b, n, c)
        return x


class DWConv(nn.Module):  # 使用卷积获取位置信息，融合在mix-ffn里
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):  # x (b, n, c)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)  # x (b, c, h, w)
        x = self.dwconv(x)  # x(b, dim, h, w)
        x = x.flatten(2).transpose(1, 2)  # x (b, n, dim)

        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化权重
        :param m:
        :return:
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2. / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x= self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = EfficientSelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.norm2 = norm_layer(dim)
        self.mlp = FFN(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.drop = nn.Dropout(drop) if drop_path > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化权重
        :param m:
        :return:
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2. / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # x (b, n, c)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x  # (b, n, c)

class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # block1
        # -----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   512, 512, 3 => 128, 128, 32 => 16384, 32
        # -----------------------------------------------#
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        cur = 0
        # -----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   16384, 32 => 16384, 32
        # -----------------------------------------------#
        self.block1 = nn.ModuleList(
            [
                Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                      norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        # block2
        #   对输入图像进行分区，并下采样
        #   128, 128, 32 => 64, 64, 64 => 4096, 64
        # -----------------------------------------------#
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[1])
        # -----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   4096, 64 => 4096, 64
        # -----------------------------------------------#
        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1]
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        # block3
        # -----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   64, 64, 64 => 32, 32, 160 => 1024, 160
        # -----------------------------------------------#
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        # -----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   1024, 160 => 1024, 160
        # -----------------------------------------------#
        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2]
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        # block4
        # -----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   32, 32, 160 => 16, 16, 256 => 256, 256
        # -----------------------------------------------#
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        # -----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   256, 256 => 256, 256
        # -----------------------------------------------#
        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3]
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # block1
        # x (b, 3, 512, 512) -> (b, embed[0], 128, 128) -> (b, 16384, embed[0])
        x, H, W = self.patch_embed1(x)
        # x (b, 16384, embed[0]) -> (b, 16384, embed[0])
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        # x (b, 16384, embed[0]) -> (b, embed[0], 128, 128)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 深拷贝
        outs.append(x)

        # block2
        # x (b, embed[0], 128, 128) -> (b, embed[1], 64, 64) -> (b, 4096, embed[1])
        x, H, W = self.patch_embed2.forward(x)
        # x (b, 4096, embed[1]) -> (b, 4096, embed[1])
        for i, blk in enumerate(self.block2):
            x = blk.forward(x, H, W)
        x = self.norm2(x)
        # x (b, 4096, embed[1]) -> (b, embed[1], 64, 64)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # block3
        # x (b, embed[1], 64, 64) -> (b, embed[2], 32, 32) -> (b, 1024, embed[2])
        x, H, W = self.patch_embed3.forward(x)
        # x (b, 1024, embed[2]) -> (b, 1024, embed[2])
        for i, blk in enumerate(self.block3):
            x = blk.forward(x, H, W)
        x = self.norm3(x)
        # x (b, 1024, embed[2]) -> (b, embed[2], 32, 32)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # block4
        # x (b, embed[2], 32, 32) -> (b, embed[3], 16, 16) -> (b, 256, embed[3])
        x, H, W = self.patch_embed4.forward(x)
        # x (b, 256, embed[3]) -> (b, 256, embed[3])
        for i, blk in enumerate(self.block4):
            x = blk.forward(x, H, W)
        x = self.norm4(x)
        # x (b, 256, embed[3]) -> (b, embed[3], 16, 16)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

class mit_b0(MixVisionTransformer):
    def __init__(self, pretrained = False):
        super().__init__(
            embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b0_backbone_weights.pth"), strict=False)

class mit_b1(MixVisionTransformer):
    def __init__(self, pretrained = False):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b1_backbone_weights.pth"), strict=False)

class mit_b2(MixVisionTransformer):
    def __init__(self, pretrained = False):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b2_backbone_weights.pth"), strict=False)

class mit_b3(MixVisionTransformer):
    def __init__(self, pretrained = False):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b3_backbone_weights.pth"), strict=False)

class mit_b4(MixVisionTransformer):
    def __init__(self, pretrained = False):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b4_backbone_weights.pth"), strict=False)

class mit_b5(MixVisionTransformer):
    def __init__(self, pretrained = False):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b5_backbone_weights.pth"), strict=False)


if __name__ == '__main__':
    model = mit_b0()
    print(model)


