"""
Model architecture definitions for BUSI breast ultrasound tumor segmentation.
Includes: Attention U-Net (Tier 1), D-SAS (Tier 2), CA-BSNet (Tier 3, corrected cross-attention).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================
# Tier 1 — Attention U-Net
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, stride=2)
        self.att4 = AttentionGate(features[3], features[3], features[3] // 2)
        self.dec4 = ConvBlock(features[3] * 2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.att3 = AttentionGate(features[2], features[2], features[2] // 2)
        self.dec3 = ConvBlock(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.att2 = AttentionGate(features[1], features[1], features[1] // 2)
        self.dec2 = ConvBlock(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.att1 = AttentionGate(features[0], features[0], features[0] // 2)
        self.dec1 = ConvBlock(features[0] * 2, features[0])

        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(g=d4, x=e4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(g=d3, x=e3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(g=d2, x=e2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(g=d1, x=e1)], dim=1))
        return self.final(d1)


# ============================================================
# Shared D-SAS backbone + modules (used by Tier 2 and Tier 3)
# ============================================================
def modify_layer4_dilation(resnet):
    for name, module in resnet.layer4.named_modules():
        if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
            module.stride = (1, 1)
            if module.kernel_size == (3, 3):
                module.dilation = (2, 2)
                module.padding = (2, 2)
    return resnet


class D_SAS_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet101(weights=None)
        resnet = modify_layer4_dilation(resnet)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.stem(x)
        f_low = self.layer1(x)
        x2 = self.layer2(f_low)
        x3 = self.layer3(x2)
        f_high = self.layer4(x3)
        return f_low, f_high


class ChannelReduce(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class SSBG(nn.Module):
    """Semantic-Spatial Boundary Gate — extracts and gates boundary features from low-level features."""

    def __init__(self, ch, edge_weight=0.1):
        super().__init__()
        self.edge_weight = edge_weight
        self.gate_dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.gate_bn = nn.BatchNorm2d(ch)
        self.gate_pw = nn.Conv2d(ch, 1, 1)
        self.edge_dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.edge_bn1 = nn.BatchNorm2d(ch)
        self.edge_pw = nn.Conv2d(ch, ch, 1)
        self.edge_bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_low):
        g = self.relu(self.gate_bn(self.gate_dw(f_low)))
        g = torch.sigmoid(self.gate_pw(g))
        f_gated = f_low * g
        e = self.relu(self.edge_bn1(self.edge_dw(f_low)))
        f_edge = self.relu(self.edge_bn2(self.edge_pw(e)))
        return f_gated + f_edge * self.edge_weight


class AAS(nn.Module):
    """Atrous multi-scale Semantic branch — dilated conv branches + global context, softmax-weighted fusion."""

    def __init__(self, in_ch=256, out_ch=256, dilations=(1, 2, 4, 6)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d)
            self.branches.append(nn.Sequential(conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, 5, 1),
        )
        self.final = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        feats = [branch(x) for branch in self.branches]
        global_feat = self.global_conv(self.global_pool(x))
        global_feat = F.interpolate(global_feat, size=(H, W), mode="bilinear", align_corners=False)
        feats.append(global_feat)
        concat = torch.cat(feats, dim=1)
        weights = torch.softmax(self.gate(concat), dim=1)
        stacked = torch.stack(feats, dim=1)
        weighted_sum = (stacked * weights.unsqueeze(2)).sum(dim=1)
        return self.dropout(self.final(weighted_sum))


class SFH(nn.Module):
    """Semantic-boundary Fusion Head — uncertainty-weighted fusion of boundary (SSBG) and semantic (AAS) features."""

    def __init__(self, ch=256, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gate_l = nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.BatchNorm2d(ch), nn.Sigmoid())
        self.gate_h = nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.BatchNorm2d(ch), nn.Sigmoid())
        self.conv_l = nn.Conv2d(ch, 1, 1)
        self.conv_h = nn.Conv2d(ch, 1, 1)
        self.out_proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

    def forward(self, f_low, f_high):
        f_high_up = F.interpolate(f_high, size=f_low.shape[2:], mode="bilinear", align_corners=False)
        g_l = self.gate_l(torch.cat([f_low, f_high_up], dim=1))
        g_h = self.gate_h(torch.cat([f_high_up, f_low], dim=1))
        f_l_hat = f_low * g_l
        f_h_hat = f_high_up * g_h
        ul = F.softplus(self.conv_l(f_l_hat)) + self.eps
        uh = F.softplus(self.conv_h(f_h_hat)) + self.eps
        fused = (f_l_hat / ul + f_h_hat / uh) / (1 / ul + 1 / uh)
        return self.out_proj(fused)


class DSASDecoder(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x, out_size):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


# ============================================================
# Tier 2 — D-SAS (reproduced base paper architecture)
# ============================================================
class DSAS(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = D_SAS_Backbone()
        self.reduce_low = ChannelReduce(256, 256)
        self.reduce_high = ChannelReduce(2048, 256)
        self.ssbg = SSBG(ch=256)
        self.aas = AAS(in_ch=256, out_ch=256)
        self.sfh = SFH(ch=256)
        self.decoder = DSASDecoder(ch=256)

    def forward(self, x):
        out_size = x.shape[2:]
        f_low, f_high = self.backbone(x)
        f_low_r = self.reduce_low(f_low)
        f_high_r = self.reduce_high(f_high)
        f_ssbg_out = self.ssbg(f_low_r)
        f_aas_out = self.aas(f_high_r)
        f_sfh_out = self.sfh(f_ssbg_out, f_aas_out)
        return self.decoder(f_sfh_out, out_size)


# ============================================================
# Tier 3 — CA-BSNet (D-SAS + cross-attention between AAS and SSBG)
# ============================================================
class CrossAttentionBlock(nn.Module):
    """True cross-attention: Q from AAS (semantic), K/V from SSBG (boundary)."""

    def __init__(self, ch=256, num_heads=4, mlp_ratio=2):
        super().__init__()
        self.norm_q = nn.LayerNorm(ch)
        self.norm_kv = nn.LayerNorm(ch)
        self.q_proj = nn.Linear(ch, ch)
        self.k_proj = nn.Linear(ch, ch)
        self.v_proj = nn.Linear(ch, ch)
        self.attn = nn.MultiheadAttention(embed_dim=ch, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.out_proj = nn.Linear(ch, ch)
        self.norm2 = nn.LayerNorm(ch)
        self.mlp = nn.Sequential(nn.Linear(ch, ch * mlp_ratio), nn.GELU(), nn.Linear(ch * mlp_ratio, ch))

    def forward(self, f_aas, f_ssbg):
        f_ssbg_small = F.interpolate(f_ssbg, size=f_aas.shape[2:], mode="bilinear", align_corners=False)
        q_tokens = f_aas.flatten(2).transpose(1, 2)
        kv_tokens = f_ssbg_small.flatten(2).transpose(1, 2)
        q = self.q_proj(self.norm_q(q_tokens))
        k = self.k_proj(self.norm_kv(kv_tokens))
        v = self.v_proj(self.norm_kv(kv_tokens))
        attn_out, attn_weights = self.attn(query=q, key=k, value=v, need_weights=True)
        tokens = q_tokens + self.out_proj(attn_out)
        tokens = tokens + self.mlp(self.norm2(tokens))
        B, N, C = tokens.shape
        H, W = f_aas.shape[2:]
        return tokens.transpose(1, 2).reshape(B, C, H, W), attn_weights


class CABSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = D_SAS_Backbone()
        self.reduce_low = ChannelReduce(256, 256)
        self.reduce_high = ChannelReduce(2048, 256)
        self.ssbg = SSBG(ch=256)
        self.aas = AAS(in_ch=256, out_ch=256)
        self.cross_attn = CrossAttentionBlock(ch=256, num_heads=4, mlp_ratio=2)
        self.sfh = SFH(ch=256)
        self.decoder = DSASDecoder(ch=256)

    def forward(self, x):
        out_size = x.shape[2:]
        f_low, f_high = self.backbone(x)
        f_low_r = self.reduce_low(f_low)
        f_high_r = self.reduce_high(f_high)
        f_ssbg_out = self.ssbg(f_low_r)
        f_aas_out = self.aas(f_high_r)
        f_ca_out, _ = self.cross_attn(f_aas_out, f_ssbg_out)
        f_sfh_out = self.sfh(f_ssbg_out, f_ca_out)
        return self.decoder(f_sfh_out, out_size)


MODEL_REGISTRY = {
    "Attention U-Net (Tier 1)": AttentionUNet,
    "D-SAS (Tier 2)": DSAS,
    "CA-BSNet (Tier 3)": CABSNet,
}
