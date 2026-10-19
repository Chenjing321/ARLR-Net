from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from tools.utils import read_targets_from_xml_list


class REBNCONV(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate,
                                 dilation=dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear',
                         align_corners=False)


class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, dilation_ratio=1):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=dilation_ratio)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=dilation_ratio)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=dilation_ratio)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=dilation_ratio)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=dilation_ratio)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=dilation_ratio)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=dilation_ratio)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx5d = self.rebnconv5d(torch.cat((_upsample_like(hx6d, hx5), hx5), 1))
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class SD2M(nn.Module):

    def __init__(self, in_channels, out_channels, shifts, kernel_size):
        super().__init__()
        self.hidden_channels = in_channels // kernel_size
        self.in_channels = in_channels
        self.shifts = shifts
        self.num_layer = 8
        delta1 = torch.tensor([
            [[[-1, 0, 0], [0, 1, 0], [0, 0, 0]]],
            [[[0, -1, 0], [0, 1, 0], [0, 0, 0]]],
            [[[0, 0, -1], [0, 1, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
        ], dtype=torch.float32)
        delta2 = torch.flip(delta1, dims=(-1, -2))
        kernel = torch.cat([delta1, delta2], dim=0)
        self.register_buffer('kernel', kernel.repeat(self.hidden_channels, 1, 1, 1))
        self.down_layer = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, 1),
            nn.BatchNorm2d(self.hidden_channels),
        )
        self.origin_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 1),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 1),
        )
        self.out_conv = nn.Conv2d(self.hidden_channels, out_channels, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=11, stride=1, padding=5)
        self.avg_pool = nn.AvgPool2d(kernel_size=11, stride=1, padding=5)

    def forward(self, cen):
        b, _, w, h = cen.shape
        cen = self.down_layer(cen)
        basises = [
            (self.max_pool(cen) - self.avg_pool(cen)).view(b, self.hidden_channels, 1, -1),
            (cen - torch.mean(cen, dim=[2, 3], keepdim=True)).view(b, self.hidden_channels, 1, -1),
        ]
        for shift in self.shifts:
            basis = F.conv2d(cen, self.kernel, stride=1, padding='same',
                             groups=self.hidden_channels, dilation=shift)
            basises.append(basis.view(b, self.hidden_channels, self.num_layer, -1))
        origin = self.origin_conv(cen).view(b, self.hidden_channels, 1, -1)
        basis = torch.cat(basises, dim=2)
        basis_norm = F.normalize(basis, dim=-1)
        weight_score = torch.matmul(origin, basis_norm.transpose(-2, -1))
        out = torch.matmul(weight_score, basis_norm).view(b, self.hidden_channels, w, h)
        return self.out_conv(out)


class SD2D(nn.Module):

    def __init__(self, dim, ratio=1):
        super().__init__()
        self.hidden_channels = dim // ratio
        kernel = torch.tensor([
            [[1, -1], [1, -1]],
            [[1, 1], [-1, -1]],
            [[1, -1], [-1, 1]],
        ], dtype=torch.float32).view(-1, 1, 2, 2)
        self.register_buffer('kernel', kernel.repeat(self.hidden_channels, 1, 1, 1))
        self.num_layer = 3
        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        self.bn = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim))
        self.origin_conv = nn.Sequential(
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 1),
        )
        self.out_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, cen):
        b, _, w, h = cen.shape
        cen = self.bn(cen)
        edge = F.conv2d(cen, self.kernel, stride=2,
                        groups=self.hidden_channels)
        edge = edge.view(b, self.hidden_channels, self.num_layer, -1)
        avg_cen = self.avg_pool(cen)
        max_basis = self.max_pool(cen) - avg_cen
        avg_basis = avg_cen - torch.mean(avg_cen, dim=[2, 3], keepdim=True)
        basis = torch.cat([
            max_basis.view(b, self.hidden_channels, 1, -1),
            avg_basis.view(b, self.hidden_channels, 1, -1),
            edge,
        ], dim=2)
        basis_norm = F.normalize(basis, dim=-1)
        origins = self.origin_conv(cen).view(b, self.hidden_channels, 1, -1)
        weight_score = torch.matmul(origins, basis_norm.transpose(-2, -1))
        out = torch.matmul(weight_score, basis_norm).view(b, self.hidden_channels, w // 2, h // 2)
        return self.out_conv(out)


def get_activation(activation_type):
    return getattr(nn, activation_type, nn.ReLU)()


class CBN(nn.Module):

    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = [CBN(in_channels, out_channels, activation)]
    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class UpBlockAttention(nn.Module):

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=False)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.sattn = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, 1),
                                   nn.Sigmoid())

    def forward(self, d, c, xin):
        d = self.up(d) + xin
        d = self.sattn(c) * d
        return self.nConvs(torch.cat([c, d], dim=1))


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = None
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class Head(nn.Module):

    def __init__(self, inpChannel, oupChannel):
        super().__init__()
        interChannel = inpChannel // 4
        self.head = nn.Conv2d(inpChannel, interChannel, 7, padding=3,
                              groups=interChannel, bias=False)
        self.out_conv = nn.Conv2d(interChannel + inpChannel, oupChannel, 1,
                                  bias=False)

    def forward(self, x):
        return self.out_conv(torch.cat([self.head(x), x], dim=1))


class SD2Core(nn.Module):

    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        in_channels = 8
        self.inc = RSU7(n_channels, in_channels, in_channels * 2,
                        dilation_ratio=1)
        self.down1 = SD2D(dim=in_channels * 2)
        self.encoder1 = ResBlock(in_channels * 2, in_channels * 2)
        self.down2 = SD2D(dim=in_channels * 2)
        self.encoder2 = ResBlock(in_channels * 2, in_channels * 4)
        self.down3 = SD2D(dim=in_channels * 4)
        self.encoder3 = ResBlock(in_channels * 4, in_channels * 8)
        self.down4 = SD2D(dim=in_channels * 8)
        self.encoder4 = ResBlock(in_channels * 8, in_channels * 8)
        self.contras1 = SD2M(in_channels * 2, in_channels * 2, [1, 2, 3], 2)
        self.contras2 = SD2M(in_channels * 2, in_channels * 2, [1, 2, 3], 2)
        self.contras3 = SD2M(in_channels * 4, in_channels * 4, [1, 2, 3], 4)
        self.contras4 = SD2M(in_channels * 8, in_channels * 8, [1, 2, 3], 8)
        self.decoder4 = UpBlockAttention(in_channels * 16, in_channels * 4, 2)
        self.decoder3 = UpBlockAttention(in_channels * 8, in_channels * 2, 2)
        self.decoder2 = UpBlockAttention(in_channels * 4, in_channels * 2, 2)
        self.decoder1 = UpBlockAttention(in_channels * 4, in_channels * 2, 2)
        self.outc = nn.Sequential(
            RSU7(in_channels * 2, in_channels, in_channels, dilation_ratio=1),
            Head(inpChannel=in_channels, oupChannel=n_classes),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.encoder1(self.down1(x1))
        x3 = self.encoder2(self.down2(x2))
        x4 = self.encoder3(self.down3(x3))
        d5 = self.encoder4(self.down4(x4))
        c1 = self.contras1(x1)
        c2 = self.contras2(x2)
        c3 = self.contras3(x3)
        c4 = self.contras4(x4)
        d4 = self.decoder4(d5, c4, x4)
        d3 = self.decoder3(d4, c3, x3)
        d2 = self.decoder2(d3, c2, x2)
        out = self.outc(self.decoder1(d2, c1, x1))
        return torch.sigmoid(out)


@MODELS.register_module()
class SD2Net(BaseModel):
    """Single-frame SDec/SD2Net adapted to the ARLR CSO contract."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 adapter_size=64,
                 output_size=33,
                 output_scale=255.0):
        super().__init__()
        self.adapter_size = adapter_size
        self.output_size = output_size
        self.output_scale = output_scale
        self.core = SD2Core(in_channels, out_channels)

    def _stack_batch_tensor(self, value, device):
        tensor = value if torch.is_tensor(value) else torch.stack(value)
        return tensor.float().to(device)

    def _format_lr_input(self, value):
        if value.dim() == 2 and value.shape[-1] == 121:
            return value.view(value.shape[0], 1, 11, 11)
        if value.dim() == 3 and value.shape[-1] == 121:
            return value.view(value.shape[0], 1, 11, 11)
        if value.dim() == 3 and value.shape[-2:] == (11, 11):
            return value.unsqueeze(1)
        if value.dim() == 4 and value.shape[-2:] == (11, 11):
            return value
        return value.view(value.shape[0], 1, 11, 11)

    def _format_hr_target(self, value):
        return value.view(value.shape[0], -1)

    def _forward_image(self, x):
        x = F.interpolate(x, size=(self.adapter_size, self.adapter_size),
                          mode='bilinear', align_corners=False)
        out = self.core(x)
        out = F.interpolate(out, size=(self.output_size, self.output_size),
                            mode='bilinear', align_corners=False)
        return out * self.output_scale

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], List[dict]]:
        mode = kwargs['mode']
        device = next(self.parameters()).device

        if mode == 'tensor':
            x = self._format_lr_input(kwargs['Phix'].float().to(device))
        else:
            Input_image = self._format_lr_input(
                self._stack_batch_tensor(kwargs['gt_img_11'], device))
            x = Input_image
            gt_key = 'gt' if 'gt' in kwargs else 'img_gt'
            batch_x = self._format_hr_target(
                self._stack_batch_tensor(kwargs[gt_key], device))
            ann_paths = kwargs.get('ann_path', None)
            file_id = kwargs.get('file_id', None)

        x_final = self._forward_image(x).reshape(x.shape[0], -1)
        if mode == 'tensor':
            return x_final
        if mode == 'predict':
            targets_GT = read_targets_from_xml_list(ann_paths) if ann_paths is not None else None
            return [{
                'x_final': x_final,
                'targets_GT': targets_GT,
                'file_id': file_id,
                'Input_image': Input_image,
                'img_gt': batch_x,
            }]
        if mode == 'loss':
            return {'loss_discrepancy': torch.mean(torch.pow(x_final - batch_x, 2))}
        raise ValueError(f'Unsupported mode: {mode}')
