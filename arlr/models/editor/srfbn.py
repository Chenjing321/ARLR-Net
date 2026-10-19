from collections import OrderedDict
import sys
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from tools.utils import read_targets_from_xml_list


def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU(inplace)
    if act_type == 'lrelu':
        return nn.LeakyReLU(slope, inplace)
    if act_type == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=slope)
    raise NotImplementedError(
        f'[ERROR] Activation layer [{act_type}] is not implemented!')


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        return nn.BatchNorm2d(n_feature)
    raise NotImplementedError(
        f'[ERROR] Normalization layer [{norm_type}] is not implemented!')


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        return nn.ReflectionPad2d(padding)
    if pad_type == 'replicate':
        return nn.ReplicationPad2d(padding)
    raise NotImplementedError(
        f'[ERROR] Padding layer [{pad_type}] is not implemented!')


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                f'[ERROR] {sys.modules[__name__]}.sequential() does not '
                'support OrderedDict')
        return args[0]

    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    return (kernel_size - 1) // 2


def ConvBlock(in_channels,
              out_channels,
              kernel_size,
              stride=1,
              dilation=1,
              bias=True,
              valid_padding=True,
              padding=0,
              act_type='relu',
              norm_type='bn',
              pad_type='zero',
              mode='CNA'):
    assert mode in ['CNA', 'NAC'], f'[ERROR] Wrong mode in [{__name__}]!'

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)

    act = activation(act_type, inplace=False) if act_type else None
    n = norm(in_channels, norm_type) if norm_type else None
    return sequential(n, act, p, conv)


def DeconvBlock(in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=1,
                bias=True,
                padding=0,
                act_type='relu',
                norm_type='bn',
                pad_type='zero',
                mode='CNA'):
    assert mode in ['CNA', 'NAC'], f'[ERROR] Wrong mode in [{__name__}]!'

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=dilation,
        bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)

    act = activation(act_type, inplace=False) if act_type else None
    n = norm(in_channels, norm_type) if norm_type else None
    return sequential(n, act, p, deconv)


def _upsample_params(upscale_factor):
    if upscale_factor not in (3, 4, 5, 7):
        raise ValueError(f'Unsupported SRFBN upscale_factor: {upscale_factor}')
    return upscale_factor, 2, upscale_factor + 4


class MeanShiftGray(nn.Conv2d):

    def __init__(self, gray_range, gray_mean=0.5, gray_std=1.0, sign=-1):
        super().__init__(1, 1, kernel_size=1)
        std = torch.Tensor([gray_std])
        self.weight.data = torch.ones(1, 1, 1, 1) / std.view(1, 1, 1, 1)
        self.bias.data = sign * gray_range * torch.Tensor([gray_mean]) / std
        for p in self.parameters():
            p.requires_grad = False


class FeedbackBlock(nn.Module):

    def __init__(self, num_features, num_groups, upscale_factor, act_type,
                 norm_type):
        super().__init__()
        stride, padding, kernel_size = _upsample_params(upscale_factor)

        self.num_groups = num_groups
        self.compress_in = ConvBlock(
            2 * num_features,
            num_features,
            kernel_size=1,
            act_type=act_type,
            norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(
                DeconvBlock(
                    num_features,
                    num_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    act_type=act_type,
                    norm_type=norm_type))
            self.downBlocks.append(
                ConvBlock(
                    num_features,
                    num_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    act_type=act_type,
                    norm_type=norm_type,
                    valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(
                    ConvBlock(
                        num_features * (idx + 1),
                        num_features,
                        kernel_size=1,
                        stride=1,
                        act_type=act_type,
                        norm_type=norm_type))
                self.downtranBlocks.append(
                    ConvBlock(
                        num_features * (idx + 1),
                        num_features,
                        kernel_size=1,
                        stride=1,
                        act_type=act_type,
                        norm_type=norm_type))

        self.compress_out = ConvBlock(
            num_groups * num_features,
            num_features,
            kernel_size=1,
            act_type=act_type,
            norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros_like(x)
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        lr_features = [x]
        hr_features = []

        for idx in range(self.num_groups):
            ld_l = torch.cat(tuple(lr_features), 1)
            if idx > 0:
                ld_l = self.uptranBlocks[idx - 1](ld_l)
            ld_h = self.upBlocks[idx](ld_l)

            hr_features.append(ld_h)
            ld_h = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                ld_h = self.downtranBlocks[idx - 1](ld_h)
            ld_l = self.downBlocks[idx](ld_h)

            lr_features.append(ld_l)

        output = torch.cat(tuple(lr_features[1:]), 1)
        output = self.compress_out(output)
        self.last_hidden = output
        return output

    def reset_state(self):
        self.should_reset = True


@MODELS.register_module()
class SRFBN(BaseModel):
    """SRFBN comparison model adapted to ARLR-Net data and metric contracts."""

    def __init__(self,
                 upscale_factor=3,
                 in_channels=1,
                 out_channels=1,
                 num_features=32,
                 num_steps=1,
                 num_groups=3,
                 act_type='prelu',
                 norm_type=None):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.num_steps = num_steps
        self.num_features = num_features

        stride, padding, kernel_size = _upsample_params(upscale_factor)

        self.sub_mean = MeanShiftGray(255)
        self.conv_in = ConvBlock(
            in_channels,
            4 * num_features,
            kernel_size=3,
            act_type=act_type,
            norm_type=norm_type)
        self.feat_in = ConvBlock(
            4 * num_features,
            num_features,
            kernel_size=1,
            act_type=act_type,
            norm_type=norm_type)
        self.block = FeedbackBlock(
            num_features, num_groups, upscale_factor, act_type, norm_type)
        self.out = DeconvBlock(
            num_features,
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_type=act_type,
            norm_type=norm_type)
        self.conv_out = ConvBlock(
            num_features,
            out_channels,
            kernel_size=3,
            act_type=None,
            norm_type=norm_type)
        self.add_mean = MeanShiftGray(255, sign=1)

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

    def _forward_image(self, x):
        self._reset_state()
        x = self.sub_mean(x)
        inter_res = F.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False)

        x = self.conv_in(x)
        x = self.feat_in(x)
        h = self.block(x)
        h = torch.add(inter_res, self.conv_out(self.out(h)))
        return self.add_mean(h)

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

        out = self._forward_image(x)
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

    def _reset_state(self):
        self.block.reset_state()
