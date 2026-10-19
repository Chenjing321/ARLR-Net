from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from tools.utils import read_targets_from_xml_list


class SEAttention(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.se(self.maxpool(x)) + self.se(self.avgpool(x)))


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([max_result, avg_result], 1)))


def gradient_1order(x, h_x=None, w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size(2)
        w_x = x.size(3)
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    return torch.sqrt(torch.pow((r - l) * 0.5, 2) +
                      torch.pow((t - b) * 0.5, 2) + 1e-12)


class Resnet1(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.layer(x) + x)


class Resnet2(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.layer1(x) + self.layer2(x))


class Resnet3(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.se = SEAttention(channel=out_channel, reduction=4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.layer2(self.se(self.layer1(x))) + x)


class Res(nn.Module):

    def __init__(self, after_channel):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, after_channel, 3, padding=1),
            nn.BatchNorm2d(after_channel),
            nn.ReLU(inplace=True),
            SEAttention(channel=after_channel, reduction=4),
            nn.Conv2d(after_channel, after_channel, 3, padding=1),
            nn.BatchNorm2d(after_channel),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2 * after_channel, after_channel, 3, padding=1),
            nn.BatchNorm2d(after_channel),
            nn.ReLU(inplace=True),
            SEAttention(channel=after_channel, reduction=4),
            nn.Conv2d(after_channel, after_channel, 3, padding=1),
            nn.BatchNorm2d(after_channel),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1):
        identity = x
        x1 = self.layer1(x1)
        return self.relu(self.layer2(torch.cat([x, x1], 1)) + identity)


class Stage(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.resnet1_1 = Resnet1(16, 16)
        self.resnet1_2 = Resnet3(16, 16)
        self.resnet1_3 = Resnet3(16, 16)
        self.res1 = Res(16)
        self.resnet2_1 = Resnet2(16, 32)
        self.resnet2_2 = Resnet3(32, 32)
        self.resnet2_3 = Resnet3(32, 32)
        self.res2 = Res(32)
        self.resnet3_1 = Resnet2(32, 64)
        self.resnet3_2 = Resnet3(64, 64)
        self.resnet3_3 = Resnet3(64, 64)
        self.res3 = Res(64)
        self.resnet4_1 = Resnet2(64, 128)
        self.resnet4_2 = Resnet3(128, 128)
        self.resnet4_3 = Resnet3(128, 128)
        self.res4 = Res(128)
        self.resnet5_1 = Resnet2(128, 256)
        self.resnet5_2 = Resnet3(256, 256)
        self.resnet5_3 = Resnet3(256, 256)
        self.res5 = Res(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, x_g):
        outs = []
        out = self.resnet1_3(self.resnet1_2(self.resnet1_1(self.layer1(x))))
        out = self.res1(out, x_g)
        outs.append(out)
        out = self.resnet2_3(self.resnet2_2(self.resnet2_1(out)))
        x1 = self.pool(x_g)
        out = self.res2(out, x1)
        outs.append(out)
        out = self.resnet3_3(self.resnet3_2(self.resnet3_1(out)))
        x2 = self.pool(x1)
        out = self.res3(out, x2)
        outs.append(out)
        out = self.resnet4_3(self.resnet4_2(self.resnet4_1(out)))
        x3 = self.pool(x2)
        out = self.res4(out, x3)
        outs.append(out)
        out = self.resnet5_3(self.resnet5_2(self.resnet5_1(out)))
        x4 = self.pool(x3)
        out = self.res5(out, x4)
        outs.append(out)
        return outs


class LCL(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer1(x)


class Sbam(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.hl_layer = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.hl_layer_2 = ChannelAttention(out_channel)
        self.ll_layer = SpatialAttention()

    def forward(self, hl, ll):
        hl = self.hl_layer(hl)
        ll_1 = ll * self.hl_layer_2(hl)
        hl_1 = hl * self.ll_layer(ll)
        return ll_1 + hl_1


class GGLCore(nn.Module):

    def __init__(self):
        super().__init__()
        self.stage = Stage()
        self.lcl5 = LCL(256, 256)
        self.lcl4 = LCL(128, 128)
        self.lcl3 = LCL(64, 64)
        self.lcl2 = LCL(32, 32)
        self.lcl1 = LCL(16, 16)
        self.sbam4 = Sbam(256, 128)
        self.sbam3 = Sbam(128, 64)
        self.sbam2 = Sbam(64, 32)
        self.sbam1 = Sbam(32, 16)
        self.layer = nn.Sequential(
            nn.Conv2d(16, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, x_g):
        outs = self.stage(x, x_g)
        out5 = self.lcl5(outs[4])
        out4 = self.lcl4(outs[3])
        out3 = self.lcl3(outs[2])
        out2 = self.lcl2(outs[1])
        out1 = self.lcl1(outs[0])
        out4 = self.sbam4(out5, out4)
        out3 = self.sbam3(out4, out3)
        out2 = self.sbam2(out3, out2)
        out1 = self.sbam1(out2, out1)
        return self.layer(out1)


@MODELS.register_module()
class GGLNet(BaseModel):
    """GGL-Net adapted to the ARLR CSO comparison contract."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 adapter_size=64,
                 output_size=33,
                 output_scale=255.0):
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError('GGLNet adapter currently expects 1 input/output channel.')
        self.adapter_size = adapter_size
        self.output_size = output_size
        self.output_scale = output_scale
        self.core = GGLCore()

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
        x3 = x.repeat(1, 3, 1, 1)
        x_g = gradient_1order(x3)
        out = self.core(x3, x_g)
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
