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
        return self.proj(x)


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), return_all_tokens=False, 
                 init_values=0, use_mean_pooling=False, masked_im_modeling=False, local_crop=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = jt.zeros((1, 1, embed_dim))
        self.pos_embed = jt.zeros((1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # SERE: projection heads required to model channel and pixel self-relations
        self.channel_sere = ChannelSERE(dim=embed_dim)
        self.pixel_sere = PixelSERE(dim=embed_dim, num_heads=num_heads)
        if local_crop:
            # SERE: using additional heads to process local crops in SERE
            self.channel_sere_local = ChannelSERE(dim=embed_dim)
            self.pixel_sere_local = PixelSERE(dim=embed_dim, num_heads=num_heads)

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
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def execute(self, x, return_all_tokens=None, mask=None, local_crop=False):
        # mim
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x)
        
        # SERE: modeling channel and pixel self-relations
        if local_crop:
            relation_channel_z, relation_channel_p = self.channel_sere_local(x[:, 1:, :], size=6) # channel self-relation
            relation_pixel_z, relation_pixel_p = self.pixel_sere_local(x[:, 1:, :]) # pixel self-relation
        else:
            relation_channel_z, relation_channel_p = self.channel_sere(x[:, 1:, :]) # channel self-relation
            relation_pixel_z, relation_pixel_p = self.pixel_sere(x[:, 1:, :]) # pixel self-relation

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))
        
        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return x, relation_channel_z, relation_channel_p, relation_pixel_z, relation_pixel_p
        return x[:, 0], relation_channel_z, relation_channel_p, relation_pixel_z, relation_pixel_p

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
    def __init__(self, in_dim, out_dim, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, **kwargs):
        super().__init__()
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        act = self._build_act(act)

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
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
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)
        
        if bottleneck_dim > 0:
            self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False), name='weight', dim=1)
            self.last_layer.weight_g.assign(jt.ones_like(self.last_layer.weight_g))
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None

        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
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


class iBOTHead(DINOHead):

    def __init__(self, *args, patch_out_dim=8192, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, 
                 shared_head=False, **kwargs):
        
        super(iBOTHead, self).__init__(*args,
                                        norm=norm,
                                        act=act,
                                        last_norm=last_norm,
                                        nlayers=nlayers,
                                        hidden_dim=hidden_dim,
                                        bottleneck_dim=bottleneck_dim,
                                        norm_last_layer=norm_last_layer, 
                                        **kwargs)

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False), name='weight', dim=1)
                self.last_layer2.weight_g.assign(jt.ones_like(self.last_layer2.weight_g))
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None

            self.last_norm2 = self._build_norm(last_norm, patch_out_dim, affine=False, **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

            self.last_norm2 = self.last_norm

    def execute(self, x):
        if len(x.shape) == 2:
            return super(iBOTHead, self).execute(x)
        
        if self.last_layer is not None:
            x = self.mlp(x)
            x = jt.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])
        
        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)
        
        return x1, x2


class ChannelSERE(nn.Module):
    """
    SERE: Projection and predictor to generate channel self-relations.
    """
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), 
            nn.BatchNorm2d(dim), 
            nn.ReLU()
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def execute(self, x, size=14):
        B, N, C = x.shape
        z = x.transpose(2, 1).view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        z = self.projection(z)
        p = self.predictor(z)
        z = z.view(B, C, -1)
        p = p.view(B, C, -1)

        z = z / size
        p = p / size
        relation_z = z @ z.transpose(-2, -1)
        relation_p = p @ p.transpose(-2, -1)

        return relation_z, relation_p


class PixelSERE(nn.Module):
    """
    SERE: Projection and predictor to generate pixel self-relations.
    """
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.projection = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), 
            nn.BatchNorm2d(dim), 
            nn.ReLU()
        )
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def execute(self, z):

        B, N, C = z.shape
        z = z.transpose(2, 1).view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        z = self.projection(z)
        p = self.predictor(z)

        p = p.view(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
        z = z.view(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)

        relation_z = (z @ z.transpose(-2, -1)) * self.scale
        relation_p = (p @ p.transpose(-2, -1)) * self.scale

        return relation_z, relation_p


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

    def execute(self, x, grid1=None, grid2=None, mask=None, return_backbone_feat=False, teacher=False,
                **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None

        idx_crops, last_size = [0], x[0].shape[-1]
        for sample in [inp.shape[-1] for inp in x]:
            if sample == last_size:
                idx_crops[-1] += 1
            else:
                idx_crops.append(idx_crops[-1] + 1)

        start_idx = 0
        relation_pixel_local, relation_channel_local = None, None
        relation_channel, relation_pixel, relation_pixel_global = None, None, None
        for end_idx in idx_crops:
            inp_x = jt.concat(x[start_idx: end_idx])

            if mask is not None:
                inp_m = jt.concat(mask[start_idx: end_idx])
                kwargs.update(dict(mask=inp_m))

            _out, _relation_channel_z, _relation_channel_p, _relation_pixel_z, _relation_pixel_p = \
                self.backbone(inp_x, local_crop=end_idx > 2, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = jt.concat((output, _out))
            
            if grid1 is not None and grid2 is not None:
                if not teacher:
                    relation_pixel = _relation_pixel_p
                    relation_channel = _relation_channel_p
                else:
                    relation_pixel = _relation_pixel_z
                    relation_channel = _relation_channel_z

                relation_pixel_global = relation_pixel

                b, h, n, _ = relation_pixel.shape  # [B H N N]
                target_n1 = grid1[0].shape[1] * grid1[0].shape[1]
                target_n2 = grid2[0].shape[1] * grid2[0].shape[1]
                relation_pixel = relation_pixel.contiguous().view(b, h * n, int(math.sqrt(n)), int(math.sqrt(n)))
                relation_pixel = (
                    jt.nn.grid_sample(
                        relation_pixel.float(),
                        jt.concat(grid2),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .view(b, h, n, target_n2)
                    .transpose(3, 2)
                )
                relation_pixel = relation_pixel.contiguous().view(
                    b, h * target_n2, int(math.sqrt(n)), int(math.sqrt(n))
                )
                relation_pixel = (
                    jt.nn.grid_sample(
                        relation_pixel.float(),
                        jt.concat(grid1),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .view(b, h, target_n2, target_n1)
                    .transpose(3, 2)
                )
            else:
                assert not teacher
                relation_pixel_local = _relation_pixel_p
                relation_channel_local = _relation_channel_p

            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_, relation_channel, relation_pixel, relation_channel_local, relation_pixel_local, relation_pixel_global
        return output_, relation_channel, relation_pixel, relation_channel_local, relation_pixel_local, relation_pixel_global
