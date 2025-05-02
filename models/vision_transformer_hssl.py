# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# iBOT: https://github.com/bytedance/ibot
# --------------------------------------------------------

import math
import jittor as jt
import jittor.nn as nn
from functools import partial
from utils import trunc_normal_, weight_norm


def interpolate(X, size=None, scale_factor=None, mode='bilinear', align_corners=False, tf_mode=False):
    if isinstance(scale_factor, int):
        scale_factor = (scale_factor, scale_factor)
    if scale_factor is not None:
        size = [int(X.shape[-2] * scale_factor[-2]), int(X.shape[-1] * scale_factor[-1])]
    if isinstance(size, int):
        size = (size, size)
    return jt.nn.resize(X, size, mode, align_corners, tf_mode)


def drop_path(x: jt.Var, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype, requires_grad=False)
    random_tensor = jt.floor(random_tensor) # binarize
    output = (x / keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training())


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def execute(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.
    Args:
        num_channels (int): The number of channels of the input Var.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def execute(self, x):
        assert x.ndim == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got Var with shape {x.shape}'
        return nn.layer_norm(x.permute(0, 2, 3,
                                       1), self.normalized_shape, self.weight,
                             self.bias, self.eps).permute(0, 3, 1, 2)


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=7, drop_path=0., short_cut=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=(kernel_size // 2), groups=dim) # depthwise conv
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.short_cut = short_cut
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def execute(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.short_cut:
            x = self.drop_path(x) + input
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
    def execute(self, x):
        x = self.proj(x)
        h, w = x.shape[2:]

        return x, h, w


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, auxiliary_depth=0, 
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), return_all_tokens=False, 
                 use_mean_pooling=False, masked_im_modeling=False, kernel_size=7):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens
        self.depth = depth
        self.auxiliary_depth = auxiliary_depth

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = jt.zeros((1, 1, embed_dim))
        self.pos_embed = jt.zeros((1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth + auxiliary_depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.conv_blocks = nn.ModuleList()
        for i in range(auxiliary_depth):
            self.conv_blocks.append(
                ConvNextBlock(
                    dim=embed_dim, kernel_size=kernel_size, drop_path=dpr[i + self.depth], short_cut=i > 0)
            )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.norm2 = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        assert not use_mean_pooling
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # masked image modeling
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = jt.zeros(1, embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return jt.concat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=None):
        B, _, w, h = x.shape
        x, fh, fw = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x), fh, fw

    def execute(self, x, return_all_tokens=None, mask=None):
        # mim
        if self.masked_im_modeling:
            assert mask is not None
            x, h, w = self.prepare_tokens(x, mask=mask)
        else:
            x, h, w = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x)
        out1 = self.norm(x)

        # conv
        B, N, C = x.shape
        x_auxiliary = x[:, 1:, :].view(B, h, w, C).permute(0, 3, 1, 2) # [B C H W]
        for conv_blk in self.conv_blocks:
            x_auxiliary = conv_blk(x_auxiliary)
        x_auxiliary_clstoken = x_auxiliary.mean(dim=-1).mean(dim=-1) # [B C]
        x_auxiliary = x_auxiliary.view(B, C, h * w).permute(0, 2, 1) # [B N C]
        x_auxiliary = jt.concat([x_auxiliary_clstoken.unsqueeze(1), x_auxiliary], dim=1)
        out2 = self.norm2(x_auxiliary)

        assert self.fc_norm is None
        
        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return out1, out2
        return out1[:, 0], out2[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
        
    def get_num_layers(self):
        return len(self.blocks)

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def execute(self, x, mask=None, return_backbone_feat=False, **kwargs):
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
            
        idx_crops, last_size = [0], x[0].shape[-1]
        for sample in [inp.shape[-1] for inp in x]:
            if sample == last_size:
                idx_crops[-1] += 1
            else:
                idx_crops.append(idx_crops[-1] + 1)
        
        outputs = []
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = jt.concat(x[start_idx: end_idx])
            if mask is not None:
                kwargs.update(dict(mask=jt.concat(mask[start_idx: end_idx])))
                
            _out_base, _out_auxiliary = self.backbone(inp_x, **kwargs)
            outputs.append((_out_base, _out_auxiliary))
            start_idx = end_idx
            
        output_base = jt.concat([out[0] for out in outputs])
        output_auxiliary = jt.concat([out[1] for out in outputs])
        
        output_base_1, output_base_2, output_auxiliary_1, output_auxiliary_2 = self.head(output_base, output_auxiliary)
        
        if return_backbone_feat:
            return output_base, output_auxiliary, output_base_1, output_base_2, output_auxiliary_1, output_auxiliary_2
        return output_base_1, output_base_2, output_auxiliary_1, output_auxiliary_2


class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    def execute(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, act='gelu', 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, **kwargs):
        super().__init__()
        assert bottleneck_dim > 0, "bottleneck_dim must be greater than 0"
        norm1 = self._build_norm(norm, hidden_dim)
        norm2 = self._build_norm(norm, hidden_dim)
        act1 = self._build_act(act)
        act2 = self._build_act(act)

        nlayers = max(nlayers, 1)
        self.mlp = self.get_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim, out_dim, norm1, act1)
        self.mlp2 = self.get_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim, out_dim, norm2, act2)
        self.apply(self._init_weights)
        
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False), name='weight', dim=1)
        self.last_layer.weight_g.assign(jt.ones_like(self.last_layer.weight_g))
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def get_mlp(self, nlayers, in_dim, bottleneck_dim, hidden_dim, out_dim, norm, act):
        if nlayers == 1:
            if bottleneck_dim > 0:
                mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            mlp = CustomSequential(*layers)
        
        return mlp

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def execute(self, x1, x2):
        x1 = self.mlp(x1)
        x2 = self.mlp2(x2)

        x1 = nn.functional.normalize(x1, dim=-1, p=2)
        x2 = nn.functional.normalize(x2, dim=-1, p=2)

        x1 = self.last_layer(x1)
        x2 = self.last_layer(x2)

        return x1, x2

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == 'syncbn':
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'ln':
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    def _build_act(self, act):
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        else:
            assert False, "unknown act type {}".format(act)
        return act


# Modified from iBOT: https://github.com/bytedance/ibot 
class HSSLHead(DINOHead):

    def __init__(self, *args, patch_out_dim=8192, norm=None, act='gelu', 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, 
                 shared_head=False, **kwargs):
        
        super(HSSLHead, self).__init__(*args,
                                        norm=norm,
                                        act=act,
                                        nlayers=nlayers,
                                        hidden_dim=hidden_dim,
                                        bottleneck_dim=bottleneck_dim,
                                        norm_last_layer=norm_last_layer, 
                                        **kwargs)
        assert shared_head
        assert bottleneck_dim > 0
        if not shared_head:
            self.last_layer2 = weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False), name='weight', dim=1)
            self.last_layer2.weight_g.assign(jt.ones_like(self.last_layer2.weight_g))
            if norm_last_layer:
                self.last_layer2.weight_g.requires_grad = False

        else:
            self.last_layer2 = self.last_layer

    def execute(self, x1, x2):

        x1 = self.mlp(x1)
        x2 = self.mlp2(x2)
        x1 = jt.normalize(x1, dim=-1, p=2)
        x2 = jt.normalize(x2, dim=-1, p=2)
        x1_1 = self.last_layer(x1[:, 0])
        x1_2 = self.last_layer2(x1[:, 1:])
        x2_1 = self.last_layer(x2[:, 0])
        x2_2 = self.last_layer2(x2[:, 1:])
        
        return x1_1, x1_2, x2_1, x2_2
