import copy

import torch
import timm
import numpy as np
import math
import pickle
import torch.nn as nn
from dataclasses import dataclass
import random
from torchvision.transforms import Resize
from torch.nn.init import trunc_normal_
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
from torch.nn import functional as F
from .modules import ModuleParallel, LayerNormParallel

class Mlp_2(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):  ### OSR
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0, # VIGOR:0.1
                 sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio + 3,
                           stride=sr_ratio,
                           padding=(sr_ratio + 3) // 2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None, ), )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x,y, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(y).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale  # attention map
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)

        return x.reshape(B, C, H, W)

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
class DAPooling(nn.Module):
    """ 门控机制融合多种池化方式 """

    def __init__(self, in_planes, ratio=4):
        super(DAPooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.gem_pool = GeM(work_with_tokens=None)  # GeM 池化
        # self.gem_pool = nn.Sequential(
        #     L2Norm(),
        #     GeM(work_with_tokens=None),
        #     nn.Flatten(1)
        # )
        self.chihua_fusion = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, 3 * in_planes, bias=False)
        )

        self.l2_norm = L2Norm(dim=1)
        # self.ln = nn.LayerNorm(in_planes)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def forward(self, x):
        b, c, _, _ = x.size()

        # 各种池化方式
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        gem_out = self.gem_pool(x).view(b, c)

        attn = self.chihua_fusion(avg_out + max_out + gem_out)
        attn = attn.view(b, c, 3)
        attn = self.sigmoid(attn)


        fused = (attn[:, :, 0].unsqueeze(-1).unsqueeze(-1) * avg_out.unsqueeze(-1).unsqueeze(-1) +
                 attn[:, :, 1].unsqueeze(-1).unsqueeze(-1) * max_out.unsqueeze(-1).unsqueeze(-1) +
                 attn[:, :, 2].unsqueeze(-1).unsqueeze(-1) * gem_out.unsqueeze(-1).unsqueeze(-1))

        # return self.ln(fused.flatten(1))
        return fused
        # return fused.flatten(1)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

class DDF(nn.Module):
    def __init__(self, dim):
        super(DDF, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B, C, 1, 1]
            nn.Flatten(),              # [B, C]
            nn.Linear(dim, dim),       #
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)

        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B, C, 1, 1]
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # x1: [B, C, H, W] (camera/street feature)
        # x2: [B, C, H, W] (fused_bev feature)
        # step 1: element-wise addition
        x_sum = x1 + x2                        # [B, C, H, W]

        # step 2: global channel-wise attention
        w = self.channel_attention(x_sum)        # [B, C]
        w = w.unsqueeze(-1).unsqueeze(-1)      # [B, C, 1, 1]

        # step 3: weighted features
        x1_w = x1 * w                          # [B, C, H, W]
        x2_w = x2 * (1 - w)                    # [B, C, H, W]

        # step 4: concatenate and fuse
        fused = torch.cat([x1_w, x2_w], dim=1)  # [B, 2C, H, W]
        fused = self.fusion_conv(fused)         # [B, C, H, W]

        # step 5: spatial-wise adaptive attention
        attn = self.spatial_attention(fused)    # [B, C, 1, 1]
        out = attn * fused                      # final output

        return out


class GT_Fusion(nn.Module):
    def __init__(self,
                 dim=1024,
                 attn_drop=0,
                 ):
        super().__init__()

        # self.proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # depth-wise
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.alpha = 0.3

        self.TB = Attention(dim=dim,num_heads=8,attn_drop=0,sr_ratio=2)
        self.CA = Attention(dim=dim,num_heads=8,attn_drop=0,sr_ratio=2)

        # self.norm = nn.LayerNorm([dim])
        self.norm = nn.GroupNorm(32, dim)

        self.ddf = DDF(dim=dim)

        self.act = nn.GELU()


        self.apply(self._init_weights)

    def forward(self, x, y,relative_pos_enc=None, coarse_location=None):


        B,C,H,W = x.shape

        street_proj = self.proj(x) + x
        bev_proj = self.proj(y) + y


        street_tb = self.TB(street_proj,street_proj,relative_pos_enc)
        street_tb = street_tb + street_proj
        street_tb = self.norm(street_tb)

        bev_tb = self.TB(bev_proj,bev_proj,relative_pos_enc)
        bev_tb = bev_tb + bev_proj
        bev_tb = self.norm(bev_tb)

        street_ca = self.CA(bev_tb,street_tb)

        fused_output = self.ddf(x,street_ca) + x

        return fused_output

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)




class TimmModel_Ours(nn.Module):

    def __init__(self,
                 model_name,
                 dim,
                 pretrained=False,
                 img_size=384,
                 **kwargs
                 ):

        super(TimmModel_Ours, self).__init__()

        self.img_size = img_size
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        elif "convnext" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,global_pool='')
            self.model.load_state_dict(torch.load("convnext_384.bin"), strict=False)

        self.fusion_module = GT_Fusion(dim=dim, attn_drop=0)

        self.da_pool = DAPooling(in_planes=dim)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # self.act = nn.GELU()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale3 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        num_heads = 8
        num_patches = 12 * 12
        sr_patches = math.ceil(12 / 1) * math.ceil(12 / 1)

        self.relative_pos_enc = nn.Parameter(torch.zeros(1, num_heads, num_patches, sr_patches), requires_grad=True)

    def get_config(self, ):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, imgq, imgr1=None, imgr2=None, groundA=None):

        if imgr2 is not None:
            ground_img_features = self.model(imgq)
            ground_bev_img_features = self.model(imgr1)
            sta_img_features = self.model(imgr2)
            if groundA is not None:
                groundA_features = self.model(groundA)
            '''
                GT_Fusion
            '''
            fusion_features = self.fusion_module(ground_img_features,ground_bev_img_features,self.relative_pos_enc)

            ground_img_features = self.gap(ground_img_features).flatten(1)
            ground_bev_img_features = self.gap(ground_bev_img_features).flatten(1)
            sta_img_features = self.da_pool(sta_img_features).flatten(1)
            if groundA is not None:
                groundA_features = self.da_pool(groundA_features).flatten(1)

            fusion_features_gate  = self.da_pool(fusion_features).flatten(1)

            if groundA is not None:
                return sta_img_features,fusion_features_gate,groundA_features ,ground_img_features,ground_bev_img_features
                # return sta_img_features, ground_img_features, groundA_features

            # return sta_img_features, fusion_features_gate

        elif imgr1 is not None:

            ground_img_features = self.model(imgq)
            ground_bev_img_features = self.model(imgr1)

            fusion_features = self.fusion_module(ground_img_features, ground_bev_img_features,self.relative_pos_enc)

            #ground_img_features = self.da_pool(ground_img_features).flatten(1)
            fusion_features = self.da_pool(fusion_features).flatten(1)

            #return ground_img_features
            return fusion_features

        else:
            image_features = self.model(imgq)

            image_features = self.da_pool(image_features).flatten(1)


            return image_features
