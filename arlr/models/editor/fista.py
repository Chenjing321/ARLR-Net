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
        raise ValueError('MAT path must point to an existing .mat file.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'MAT file does not exist: {path}')

    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            'Fista needs scipy to read MATLAB .mat sampling matrices. '
            'Please install scipy in the active environment.') from exc

    data = sio.loadmat(path)
    for key in candidate_keys:
        if key in data:
            return data[key]

    available = [key for key in data.keys() if not key.startswith('__')]
    raise KeyError(
        f'None of keys {candidate_keys} were found in {path}. '
        f'Available keys: {available}')


def initialize_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, 0, 0.01)
            init.constant_(module.bias, 0)


class Fista_BasicBlock(nn.Module):

    def __init__(self, features=32):
        super().__init__()
        self.Sp = nn.Softplus()

        self.conv_D = nn.Conv2d(1, features, (3, 3), stride=1, padding=1)
        self.conv1_forward = nn.Conv2d(
            features, features, (3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(
            features, features, (3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv2d(
            features, features, (3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv2d(
            features, features, (3, 3), stride=1, padding=1)

        self.conv1_backward = nn.Conv2d(
            features, features, (3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(
            features, features, (3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv2d(
            features, features, (3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv2d(
            features, features, (3, 3), stride=1, padding=1)
        self.conv_G = nn.Conv2d(features, 1, (3, 3), stride=1, padding=1)

    def forward(self, x, PhiTPhi, PhiTb, lambda_step, soft_thr):
        x = x - self.Sp(lambda_step) * torch.mm(x, PhiTPhi)
        x = x + self.Sp(lambda_step) * PhiTb

        x_input = x.view(-1, 1, 33, 33)

        x_D = self.conv_D(x_input)

        x = self.conv1_forward(x_D)
        x = F.relu(x)
        # Keep the original FISTA-Net comparison block structure.
        # self.conv2_forward and self.conv3_forward are intentionally unused.
        x_forward = self.conv4_forward(x)

        x_st = torch.mul(
            torch.sign(x_forward),
            F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))

        x = self.conv1_backward(x_st)
        x = F.relu(x)
        # Keep the original FISTA-Net comparison block structure.
        # self.conv2_backward and self.conv3_backward are intentionally unused.
        x_backward = self.conv4_backward(x)

        x_G = self.conv_G(x_backward)

        x_pred = F.relu(x_input + x_G)
        x_pred = x_pred.view(-1, 1089)

        x = self.conv1_backward(x_forward)
        x = F.relu(x)
        x_D_est = self.conv4_backward(x)
        symloss = x_D_est - x_D

        return [x_pred, symloss, x_st]


def l1_loss(pred, target, l1_weight):
    err = torch.mean(torch.abs(pred - target))
    return l1_weight * err


@MODELS.register_module()
class Fista(BaseModel):
    """FISTA-Net comparison model adapted to the ARLR-Net data contract."""

    def __init__(self,
                 LayerNo=7,
                 Phi_data_Name='data/sampling_matrix/a_phi_0_3.mat',
                 Qinit_Name='data/initial_matrix/Q_3.mat'):
        super().__init__()

        Phi = _load_mat_array(Phi_data_Name, ('phi', 'Phi'))
        Qinit = _load_mat_array(Qinit_Name, ('Qinit', 'qinit', 'Q', 'q'))

        self.register_buffer('Phi', torch.from_numpy(Phi).float())
        self.register_buffer('Qinit', torch.from_numpy(Qinit).float())
        if self.Phi.shape != (121, 1089):
            raise ValueError(
                f'Phi must have shape [121, 1089], got {tuple(self.Phi.shape)}')
        if self.Qinit.shape != (1089, 121):
            raise ValueError(
                f'Qinit must have shape [1089, 121], got {tuple(self.Qinit.shape)}')

        self.LayerNo = LayerNo

        onelayer = []
        self.bb = Fista_BasicBlock()
        for _ in range(LayerNo):
            onelayer.append(self.bb)

        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)

        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))

        self.Sp = nn.Softplus()

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

        xold = x
        y = xold
        layers_sym = []
        layers_st = []
        intermediate = [x]

        for i in range(self.LayerNo):
            theta_ = self.w_theta * i + self.b_theta
            mu_ = self.w_mu * i + self.b_mu

            xnew, layer_sym, layer_st = self.fcs[i](
                y, PhiTPhi, PhiTb, mu_, theta_)
            intermediate.append(xnew)
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) -
                    self.Sp(self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
            y = xnew + rho_ * (xnew - xold)
            xold = xnew

            layers_st.append(layer_st)
            layers_sym.append(layer_sym)

        x_final = xnew
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
            loss_discrepancy = loss_discrepancy + l1_loss(x_final, batch_x, 0.1)

            loss_constraint = x_final.new_tensor(0.0)
            for layer_sym in layers_sym:
                loss_constraint = loss_constraint + torch.mean(torch.pow(layer_sym, 2))

            sparsity_constraint = x_final.new_tensor(0.0)
            for layer_st in layers_st:
                sparsity_constraint = sparsity_constraint + torch.mean(torch.abs(layer_st))

            loss = (loss_discrepancy + 0.01 * loss_constraint +
                    0.001 * sparsity_constraint)
            return {'loss': loss}
        raise ValueError(f'Unsupported mode: {mode}')
