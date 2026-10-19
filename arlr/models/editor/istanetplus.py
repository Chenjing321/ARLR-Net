import os
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from torch.nn import init

from tools.utils import read_targets_from_xml_list


def _load_mat_array(path, candidate_keys):
    if path is None:
        raise ValueError('Qinit_Name must point to an existing .mat file.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'MAT file does not exist: {path}')

    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            'ISTANetplus needs scipy to read MATLAB .mat sampling matrices. '
            'Please install scipy in the active environment.') from exc

    data = sio.loadmat(path)
    for key in candidate_keys:
        if key in data:
            return data[key]

    available = [key for key in data.keys() if not key.startswith('__')]
    raise KeyError(
        f'None of keys {candidate_keys} were found in {path}. '
        f'Available keys: {available}')


class BasicBlock(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        c = kwargs['c']
        self.c = c

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv_D = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(
            init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 11 * self.c, 11 * self.c)

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward),
                      F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)
        x_pred = x_input + x_G
        x_pred = x_pred.view(-1, 11 * self.c * 11 * self.c)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


@MODELS.register_module()
class ISTANetplus(BaseModel):
    """ISTA-Net+ comparison model adapted to the ARLR-Net data contract."""

    def __init__(self, LayerNo, Phi_data_Name, Qinit_Name, c=3):
        super().__init__()
        self.c = c
        self.output_side = 11 * self.c
        self.output_dim = self.output_side * self.output_side

        Phi_input = _load_mat_array(Phi_data_Name, ('phi', 'Phi'))
        Qinit = _load_mat_array(Qinit_Name, ('Qinit', 'qinit', 'Q', 'q'))

        self.register_buffer('Phi', torch.from_numpy(Phi_input).float())
        self.register_buffer('Qinit', torch.from_numpy(Qinit).float())
        if self.Phi.shape != (121, self.output_dim):
            raise ValueError(
                f'Phi must have shape [121, {self.output_dim}], got '
                f'{tuple(self.Phi.shape)}')
        if self.Qinit.shape != (self.output_dim, 121):
            raise ValueError(
                f'Qinit must have shape [{self.output_dim}, 121], got '
                f'{tuple(self.Qinit.shape)}')

        self.LayerNo = LayerNo
        self.fcs = nn.ModuleList([BasicBlock(c=c) for _ in range(LayerNo)])

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

    def _format_phix(self, value):
        if value.dim() == 4 and value.shape[-2:] == (11, 11):
            return value.view(value.shape[0], -1)
        if value.dim() == 3 and value.shape[-2:] == (11, 11):
            return value.view(value.shape[0], -1)
        if value.dim() == 3 and value.shape[-1] == 121:
            return value.view(value.shape[0], -1)
        return value.view(value.shape[0], -1)

    def _check_shapes(self, Input_image, Phix, x_final, batch_x=None):
        if Input_image.shape[1:] != (1, 11, 11):
            raise ValueError(
                f'Input_image must have shape [B, 1, 11, 11], got '
                f'{tuple(Input_image.shape)}')
        if Phix.dim() != 2 or Phix.shape[1] != 121:
            raise ValueError(f'Phix must have shape [B, 121], got {tuple(Phix.shape)}')
        if x_final.dim() != 2 or x_final.shape[1] != self.output_dim:
            raise ValueError(
                f'x_final must have shape [B, {self.output_dim}], got '
                f'{tuple(x_final.shape)}')
        if batch_x is not None and (
                batch_x.dim() != 2 or batch_x.shape[1] != self.output_dim):
            raise ValueError(
                f'img_gt must have shape [B, {self.output_dim}], got '
                f'{tuple(batch_x.shape)}')

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], List[dict], torch.Tensor]:
        mode = kwargs['mode']
        device = next(self.parameters()).device

        if mode == 'tensor':
            raw_input = kwargs['Phix'].float().to(device)
            Input_image = self._format_lr_input(raw_input)
            Phix = self._format_phix(Input_image)
            batch_x = None
        else:
            Input_image = self._format_lr_input(
                self._stack_batch_tensor(kwargs['gt_img_11'], device))
            Phix = self._format_phix(Input_image)

            gt_key = 'gt' if 'gt' in kwargs else 'img_gt'
            batch_x = self._format_hr_target(
                self._stack_batch_tensor(kwargs[gt_key], device))
            ann_paths = kwargs.get('ann_path', None)
            file_id = kwargs.get('file_id', None)

        PhiTPhi = torch.mm(torch.transpose(self.Phi, 0, 1), self.Phi)
        PhiTb = torch.mm(Phix, self.Phi)
        x = torch.mm(Phix, torch.transpose(self.Qinit, 0, 1))

        layers_sym = []
        intermediate = [x]

        for layer in self.fcs:
            x, layer_sym = layer(x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
            intermediate.append(x)

        x_final = x
        self._check_shapes(Input_image, Phix, x_final, batch_x)

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
            loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
            for k in range(self.LayerNo - 1):
                loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
            return {
                'loss_discrepancy': loss_discrepancy,
                '0.01 * loss_constraint': x_final.new_tensor(0.01) * loss_constraint,
            }
        raise ValueError(f'Unsupported mode: {mode}')
