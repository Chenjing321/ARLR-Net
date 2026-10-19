import os
from typing import Dict, List, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from arlr.models.blocks import *
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
            'DISTA needs scipy to read MATLAB .mat sampling matrices. '
            'Please install scipy in the active environment.') from exc

    data = sio.loadmat(path)
    for key in candidate_keys:
        if key in data:
            return data[key]

    available = [key for key in data.keys() if not key.startswith('__')]
    raise KeyError(
        f'None of keys {candidate_keys} were found in {path}. '
        f'Available keys: {available}')


@MODELS.register_module()
class DISTA(BaseModel):
    def __init__(self,
                 LayerNo,
                 Phi_data_Name,
                 Qinit_Name,
                 block='DIST_BasicBlock',
                 lambda_weight=0.5,
                 c=3,
                 noise_level=0,
                 is_mat=False):
        super(DISTA, self).__init__()
        self.noise_level = noise_level
        self.c = c
        self.is_mat = is_mat
        self.side = 11 * c
        self.N = self.side * self.side

        Phi = _load_mat_array(Phi_data_Name, ('phi', 'Phi'))
        Qinit = _load_mat_array(Qinit_Name, ('Qinit', 'qinit', 'Q', 'q'))

        expected_phi_shape = (121, self.N)
        expected_qinit_shape = (self.N, 121)
        if Phi.shape != expected_phi_shape:
            raise ValueError(
                f'For c={c}, Phi must have shape {expected_phi_shape}, '
                f'got {tuple(Phi.shape)} from {Phi_data_Name}.')
        if Qinit.shape != expected_qinit_shape:
            raise ValueError(
                f'For c={c}, Qinit must have shape {expected_qinit_shape}, '
                f'got {tuple(Qinit.shape)} from {Qinit_Name}.')

        self.register_buffer('Phi', torch.from_numpy(Phi).float())
        self.register_buffer('Qinit', torch.from_numpy(Qinit).float())

        module_dict = globals()
        if block not in module_dict:
            raise KeyError(
                f'Missing DISTA block {block}. Do not substitute it with '
                'ARLR_BasicBlock.')
        BasicBlock = module_dict[block]

        self.LayerNo = LayerNo
        self.fcs = nn.ModuleList([
            BasicBlock(lambda_weight=lambda_weight, c=c)
            for _ in range(LayerNo)
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
                f'{tuple(Input_image.shape)}.')
        if Phix.dim() != 2 or Phix.shape[1] != 121:
            raise ValueError(f'Phix must have shape [B, 121], got {tuple(Phix.shape)}.')
        if x_final.dim() != 2 or x_final.shape[1] != self.N:
            raise ValueError(
                f'For c={self.c}, x_final must have shape [B, {self.N}], '
                f'got {tuple(x_final.shape)}.')
        if batch_x is not None and (
                batch_x.dim() != 2 or batch_x.shape[1] != self.N):
            raise ValueError(
                f'For c={self.c}, img_gt must have shape [B, {self.N}], '
                f'got {tuple(batch_x.shape)}.')

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], List[dict], torch.Tensor]:
        mode = kwargs['mode']
        device = self.Phi.device
        Phi = self.Phi
        Qinit = self.Qinit

        if mode == 'tensor':
            Input_image = self._format_lr_input(kwargs['Phix'].float().to(device))
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
            count = kwargs.get('count', None)

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        layers_sym = []

        for i in range(self.LayerNo):
            x, layer_sym = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x
        self._check_shapes(Input_image, Phix, x_final, batch_x)

        if mode == 'tensor':
            return x_final
        if mode == 'predict':
            targets_GT = read_targets_from_xml_list(ann_paths, self.is_mat)
            output = {
                'x_final': x_final,
                'targets_GT': targets_GT,
                'file_id': file_id,
                'Input_image': Input_image,
                'img_gt': batch_x,
            }
            if count is not None:
                output['count'] = count
            return [output]
        if mode == 'loss':
            loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))
            loss_constraint = x_final.new_tensor(0.0)
            for layer_sym in layers_sym:
                loss_constraint = loss_constraint + torch.mean(
                    torch.pow(layer_sym, 2))
            gamma = x_final.new_tensor(0.01)
            return {
                'loss_discrepancy': loss_discrepancy,
                'loss_constraint': gamma * loss_constraint,
            }
        raise ValueError(f'Unsupported mode: {mode}')
