import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from tools.utils import read_targets_from_xml_list


class UpsampleBlock(nn.Module):

    def __init__(self, n_channels, scale, multi_scale, wn, group=1):
        super().__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, wn=wn, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, wn=wn, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, wn=wn, group=group)
        else:
            self.up = _UpsampleBlock(
                n_channels, scale=scale, wn=wn, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            if scale == 3:
                return self.up3(x)
            if scale == 4:
                return self.up4(x)
            raise ValueError(f'Unsupported FENet scale: {scale}')
        return self.up(x)


class _UpsampleBlock(nn.Module):

    def __init__(self, n_channels, scale, wn, group=1):
        super().__init__()

        modules = []

        if scale in (2, 4, 8):
            for _ in range(int(math.log(scale, 2))):
                modules += [
                    wn(
                        nn.Conv2d(
                            n_channels,
                            4 * n_channels,
                            3,
                            1,
                            1,
                            groups=group)),
                    nn.ReLU(inplace=True),
                    nn.PixelShuffle(2),
                ]
        elif scale == 3:
            modules += [
                wn(
                    nn.Conv2d(
                        n_channels,
                        9 * n_channels,
                        3,
                        1,
                        1,
                        groups=group)),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(3),
            ]
        elif scale == 5:
            modules += [
                wn(
                    nn.Conv2d(
                        n_channels,
                        25 * n_channels,
                        3,
                        1,
                        1,
                        groups=group)),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(5),
            ]
        else:
            raise ValueError(f'Unsupported FENet scale: {scale}')

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)


class BasicConv2d(nn.Module):

    def __init__(self, wn, in_planes, out_planes, kernel_size, stride,
                 padding=0):
        super().__init__()
        self.conv = wn(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class FE(nn.Module):
    """Frequency Enhancement operation used by FENet."""

    def __init__(self, wn, in_channels, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.k2 = wn(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False))
        self.k3 = wn(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False))
        self.k4 = wn(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False))

    def forward(self, x):
        h1 = F.interpolate(
            self.pool(x), (x.size(-2), x.size(-1)), mode='nearest')
        h2 = x - h1
        f2 = torch.sigmoid(torch.add(self.k2(h2), x))
        out = torch.mul(self.k3(x), f2)
        return self.k4(out)


class FEB(nn.Module):
    """Frequency-Based Enhancement Block used by FENet."""

    def __init__(self, wn, in_channels, out_channels):
        super().__init__()
        channels = in_channels // 2
        self.path_1 = wn(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False))
        self.path_2 = wn(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.k1 = wn(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False))
        self.HConv = FE(wn, channels, channels)
        self.conv = wn(
            nn.Conv2d(
                channels * 2, out_channels, kernel_size=1, bias=False))

    def forward(self, x):
        path_1 = self.relu(self.path_1(x))
        path_1 = self.relu(self.k1(path_1))

        path_2 = self.relu(self.path_2(x))
        path_2 = self.relu(self.HConv(path_2))

        output = self.conv(torch.cat([path_1, path_2], dim=1))
        return output + x


@MODELS.register_module()
class FENet(BaseModel):
    """Frequency-Based Enhancement Network for CSO SR comparison."""

    def __init__(self, scale=3, lst_channels=1, group=4, n_blocks=12):
        super().__init__()

        wn = torch.nn.utils.weight_norm
        self.n_blocks = n_blocks
        self.scale = scale

        self.entry_1 = wn(nn.Conv2d(lst_channels, 64, 3, 1, 1))
        self.body = nn.Sequential(
            *[FEB(wn, 64, 64) for _ in range(self.n_blocks)])
        self.reduction = BasicConv2d(wn, 64 * (self.n_blocks + 1), 64, 1, 1,
                                     0)
        self.upscample = UpsampleBlock(
            64, scale=scale, multi_scale=False, wn=wn, group=group)
        self.exit = wn(nn.Conv2d(64, lst_channels, 3, 1, 1))

    def _stack_batch_tensor(self, value, device):
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.stack(value)
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

        if x.shape[-3:] != (1, 11, 11):
            raise ValueError(
                f'Input_image must have shape [B, 1, 11, 11], got '
                f'{tuple(x.shape)}')

        res = x
        x = self.entry_1(x)

        out_blocks = [x]
        for i in range(self.n_blocks):
            x = self.body[i](x)
            out_blocks.append(x)

        output = self.reduction(torch.cat(out_blocks, 1))
        output = output + x
        output = self.upscample(output, scale=self.scale)
        output = self.exit(output)

        skip = F.interpolate(
            res,
            (x.size(-2) * self.scale, x.size(-1) * self.scale),
            mode='bicubic',
            align_corners=False)
        out = skip + output
        x_final = out.reshape(out.shape[0], -1)

        if mode == 'tensor':
            return x_final
        if mode == 'predict':
            targets_GT = read_targets_from_xml_list(ann_paths)
            return [{
                'x_final': x_final,
                'targets_GT': targets_GT,
                'file_id': file_id,
                'Input_image': Input_image,
                'img_gt': batch_x,
            }]
        if mode == 'loss':
            loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))
            return {'loss_discrepancy': loss_discrepancy}
        raise ValueError(f'Unsupported mode: {mode}')
