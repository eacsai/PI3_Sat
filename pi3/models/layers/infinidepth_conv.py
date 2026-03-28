import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_to_multiple_of_4(n):
    return round(n / 4) * 4

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, output_act='elu'):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers += [nn.Linear(lastv, hidden), nn.ReLU()]
            lastv = hidden

        if out_dim is not None:
            layers.append(nn.Linear(lastv, out_dim))
            act = {
                "sigmoid": nn.Sigmoid(),
                "relu": nn.ReLU(),
                "elu": nn.ELU(),
            }.get(output_act, nn.Identity())
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class ImplicitHead(nn.Module):
    """
    Implicit head that fuses DINOv3 semantic features and BasicEncoder low-level features.

    Args:
        hidden_dim: DINOv2 feature dimension (e.g., 1024)
        basic_dim: BasicEncoder feature dimension (e.g., 128)
        fusion_type: Feature fusion strategy
            - "concat": Simple concatenation
            - "cross_attn": Cross-attention between features
            - "gated": Gated fusion with learnable weights
        out_dim: Output dimension (1 for depth)
        hidden_list: MLP hidden layer dimensions
    """
    def __init__(
            self,
            hidden_dim,  # 1024 for DINOv2
            basic_dim=128,  # BasicEncoder output dim
            fusion_type="gated",  # concat, gated
            out_dim=1,
            hidden_list=[1024, 256, 32],
            ):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.basic_dim = basic_dim
        self.fusion_type = fusion_type

        # Determine input dimension based on fusion type
        if fusion_type == "concat":
            # Simple concatenation
            in_channels = hidden_dim + basic_dim
        elif fusion_type == "gated":
            # Gated fusion with learnable weights
            self.gate_proj = nn.Linear(basic_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            in_channels = hidden_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.out_layer = MLP(
            in_dim=in_channels,
            out_dim=out_dim,
            hidden_list=hidden_list,
            output_act='identity'
        )

    def _encode_feat(self, features, patch_h, patch_w):
        """Extract DINOv3 feature map."""
        out_feat = features.permute(0, 2, 1).reshape((features.shape[0], features.shape[-1], patch_h, patch_w))
        return out_feat

    def _decode_dpt(self, feat, basic_feat, coord):
        """
        Query features at given coordinates and fuse them.

        Args:
            feat: DINOv3 feature map [B, hidden_dim, H_dino, W_dino]
            basic_feat: BasicEncoder feature map [B, basic_dim, H_basic, W_basic]
            coord: Query coordinates [B, N, 2] in (u, v) format, range [0, 1]

        Returns:
            pred: Predicted output [B, N, out_dim]
        """
        # [0, 1] → [-1, 1] for grid_sample; coord is (u, v) = (x, y), matching grid_sample convention
        coord_ = coord * 2.0 - 1.0
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

        # Sample DINOv3 features at query coordinates
        q_feat_dino = F.grid_sample(
            feat, coord_.unsqueeze(1),
            mode='bilinear', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)  # [B, N, hidden_dim]

        # Sample BasicEncoder features at query coordinates (if available)
        if basic_feat is not None:
            q_feat_basic = F.grid_sample(
                basic_feat, coord_.unsqueeze(1),
                mode='bilinear', align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)  # [B, N, basic_dim]

            # Fuse features based on fusion type
            q_feat_fused = self._fuse_features(q_feat_dino, q_feat_basic)
        else:
            # If no basic features, use only DINOv3
            q_feat_fused = q_feat_dino

        # Predict depth
        pred = self.out_layer(q_feat_fused)
        return pred

    def _fuse_features(self, feat_dino, feat_basic):
        """
        Fuse DINOv3 and BasicEncoder features.

        Args:
            feat_dino: [B, N, hidden_dim]
            feat_basic: [B, N, basic_dim]

        Returns:
            fused_feat: [B, N, fused_dim]
        """
        if self.fusion_type == "concat":
            # Simple concatenation
            return torch.cat([feat_dino, feat_basic], dim=-1)

        elif self.fusion_type == "gated":
            # Gated fusion with learnable weights
            feat_basic_proj = self.gate_proj(feat_basic)  # [B, N, hidden_dim]
            gate_input = torch.cat([feat_dino, feat_basic_proj], dim=-1)
            gate_weights = self.gate(gate_input)  # [B, N, hidden_dim]
            return gate_weights * feat_dino + (1 - gate_weights) * feat_basic_proj

    def forward(self, features, basic_feat, patch_h, patch_w, coords):
        """
        Forward pass.

        Args:
            features: DINOv3 features from backbone [B, patch_h*patch_w, hidden_dim]
            basic_feat: BasicEncoder features [B, basic_dim, H/4, W/4]
            patch_h, patch_w: DINOv3 feature map spatial size
            coords: Query coordinates [B, N, 2] in (u, v) format, range [0, 1]

        Returns:
            output: Fused embeddings [B, N, out_dim]
        """
        # Extract DINOv3 feature map
        feat = self._encode_feat(features, patch_h, patch_w)  # [B, hidden_dim, H/14, W/14]

        # Query and fuse features at coordinates
        dpt_pred = self._decode_dpt(feat, basic_feat, coords)

        return dpt_pred


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

        
class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=256, stride=4):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )

        a = _bilinear_intepolate(a)
        b = _bilinear_intepolate(b)
        c = _bilinear_intepolate(c)
        d = _bilinear_intepolate(d)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x