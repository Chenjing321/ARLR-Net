import os
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from tools.utils import read_targets_from_xml_list


def _load_mat_array(path, candidate_keys):
    if path is None:
        raise ValueError('MAT path must point to an existing .mat file.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'MAT file does not exist: {path}')

    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            'LISTA needs scipy to read MATLAB .mat sampling matrices. '
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

    def __init__(self, Phi, Qinit, theta):
        super().__init__()
        self._N = Phi.shape[1]
        self._A = Phi.astype(np.float32)

        W = np.eye(self._N, dtype=np.float32) - np.matmul(Qinit, self._A)
        self.Bs = nn.Parameter(torch.from_numpy(Qinit).float(), requires_grad=True)
        self.soft_thr = nn.Parameter(
            torch.from_numpy(theta).float(), requires_grad=True)
        self.Ws = nn.Parameter(torch.from_numpy(W).float(), requires_grad=True)

    def forward(self, xh, y):
        By = torch.matmul(self.Bs, y)
        Wxh = torch.matmul(self.Ws, xh)
        xh = soft_threshold(Wxh + By, self.soft_thr)

        return xh


def soft_threshold(input_tensor, theta):
    theta = torch.clamp(theta, min=0.0)

    return torch.sign(input_tensor) * torch.maximum(
        torch.abs(input_tensor) - theta,
        torch.zeros_like(input_tensor),
    )


@MODELS.register_module()
class LISTA(BaseModel):
    """LISTA comparison model adapted to the ARLR-Net data contract."""

    def __init__(self,
                 LayerNo=16,
                 Phi_data_Name='data/sampling_matrix/a_phi_0_3.mat',
                 Qinit_Name='data/initial_matrix/Q_3.mat',
                 lam=0.4):
        super().__init__()

        Phi = _load_mat_array(Phi_data_Name, ('phi', 'Phi')).astype(np.float32)
        Qinit = _load_mat_array(
            Qinit_Name, ('Qinit', 'qinit', 'Q', 'q')).astype(np.float32)

        if Phi.shape != (121, 1089):
            raise ValueError(f'Phi must have shape [121, 1089], got {Phi.shape}')
        if Qinit.shape != (1089, 121):
            raise ValueError(
                f'Qinit must have shape [1089, 121], got {Qinit.shape}')

        self.register_buffer('Phi', torch.from_numpy(Phi).float())
        self.register_buffer('Qinit_tensor', torch.from_numpy(Qinit).float())

        self._T = LayerNo
        self.M = Phi.shape[0]
        self._N = Phi.shape[1]
        self.LayerNo = LayerNo

        self._A = Phi
        self._lam = lam
        self._scale = 1.001 * np.linalg.norm(self._A, ord=2) ** 2
        theta_value = np.float32(self._lam / self._scale)
        self._theta = np.ones((self._N, 1), dtype=np.float32) * theta_value

        self.fcs = nn.ModuleList([
            BasicBlock(Phi, Qinit, self._theta) for _ in range(self.LayerNo)
        ])

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
        if x_final.dim() != 2 or x_final.shape[1] != 1089:
            raise ValueError(
                f'x_final must have shape [B, 1089], got {tuple(x_final.shape)}')
        if batch_x is not None and (batch_x.dim() != 2 or batch_x.shape[1] != 1089):
            raise ValueError(
                f'img_gt must have shape [B, 1089], got {tuple(batch_x.shape)}')

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], List[dict], torch.Tensor]:
        mode = kwargs['mode']
        device = self.Phi.device

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

        y = Phix.t()
        if y.dim() != 2 or y.shape[0] != 121:
            raise ValueError(f'y must have shape [121, B], got {tuple(y.shape)}')

        xh = torch.matmul(self.Qinit_tensor, y)
        if xh.dim() != 2 or xh.shape[0] != 1089:
            raise ValueError(f'xh must have shape [1089, B], got {tuple(xh.shape)}')

        for layer in self.fcs:
            xh = layer(xh, y)

        x_final = xh.t()
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
            return {'loss_discrepancy': loss_discrepancy}
        raise ValueError(f'Unsupported mode: {mode}')
