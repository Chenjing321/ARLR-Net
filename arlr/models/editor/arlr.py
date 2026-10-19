import torch
import torch.nn as nn
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
from arlr.models.blocks import *
from tools.utils import read_targets_from_xml_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SubpixelOffsetHead(nn.Module):
    def __init__(self, hidden_dim=16, scale=0.5):
        super().__init__()
        self.scale = scale
        self.offset = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(self, x):
        local_mean = nn.functional.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1)
        local_contrast = x - local_mean
        offset_input = torch.cat([x, local_contrast], dim=1)
        return torch.tanh(self.offset(offset_input)) * self.scale


@MODELS.register_module()
class ARLR(BaseModel):
    def __init__(self,
                 LayerNo,
                 Phi_data_Name,
                 Qinit_Name,
                 block="ARLR_BasicBlock",
                 lambda_weight=0.7,
                 threshold_module='AdaptiveLSKThreshold',
                 dynamic_conv_module='CorrectedDynamicKernelMixture',
                 c=3,
                 noise_level=0,
                 is_mat=False,
                 subpixel_offset=True,
                 subpixel_offset_loss_weight=0.05,
                 subpixel_offset_scale=0.5,
                 subpixel_offset_beta=0.05,
                 subpixel_offset_detach=False):
        super(ARLR, self).__init__()
        self.noise_level = noise_level
        self.c = c
        self.is_mat = is_mat
        self.threshold_module = threshold_module
        self.dynamic_conv_module = dynamic_conv_module
        self.subpixel_offset = subpixel_offset
        self.subpixel_offset_loss_weight = subpixel_offset_loss_weight
        self.subpixel_offset_beta = subpixel_offset_beta
        self.subpixel_offset_detach = subpixel_offset_detach
        self.use_subpixel_offset = (
            subpixel_offset or subpixel_offset_loss_weight > 0)

        Phi_data = sio.loadmat(Phi_data_Name)
        Phi = Phi_data['phi']
        self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        module_dict = globals()
        BasicBlock = module_dict[block]

        onelayer = []
        self.LayerNo = LayerNo
        for _ in range(LayerNo):
            onelayer.append(BasicBlock(
                lambda_weight=lambda_weight,
                threshold_module=threshold_module,
                dynamic_conv_module=dynamic_conv_module,
                c=c))

        self.fcs = nn.ModuleList(onelayer)
        if self.use_subpixel_offset:
            self.subpixel_offset_head = SubpixelOffsetHead(
                scale=subpixel_offset_scale)

    def build_subpixel_offset_targets(self, ann_paths, height, width):
        offset_target = torch.zeros(
            (len(ann_paths), 2, height, width),
            dtype=torch.float32,
            device=device)
        offset_count = torch.zeros(
            (len(ann_paths), 1, height, width),
            dtype=torch.float32,
            device=device)
        center = (self.c - 1) // 2

        targets = read_targets_from_xml_list(ann_paths, self.is_mat)
        for batch_idx, ann in enumerate(targets):
            for row_coord, col_coord, _ in ann:
                row_hr = self.c * row_coord + center
                col_hr = self.c * col_coord + center
                row = int(round(row_hr))
                col = int(round(col_hr))
                if 0 <= row < height and 0 <= col < width:
                    offset_target[batch_idx, 0, row, col] += row_hr - row
                    offset_target[batch_idx, 1, row, col] += col_hr - col
                    offset_count[batch_idx, 0, row, col] += 1.0

        offset_target = offset_target / offset_count.clamp_min(1.0)
        valid_mask = offset_count > 0
        return offset_target, valid_mask

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:
        mode = kwargs['mode']
        Phi = self.Phi
        Qinit = self.Qinit

        if mode == 'tensor':
            Phix = kwargs["Phix"]
        else:
            raw_img = torch.stack(kwargs["gt_img_11"]).to(device)
            if raw_img.dim() == 2 and raw_img.shape[-1] == 121:
                raw_img = raw_img.view(raw_img.shape[0], 1, 11, 11)
            if raw_img.dim() == 3 and raw_img.shape[-2:] == (11, 11):
                raw_img = raw_img.unsqueeze(1)
            if raw_img.dim() == 3 and raw_img.shape[-1] == 121:
                raw_img = raw_img.view(raw_img.shape[0], 1, 11, 11)
            Input_image = raw_img

            if raw_img.dim() == 4:
                Phix = raw_img.view(raw_img.shape[0], -1)
            else:
                Phix = raw_img.squeeze(dim=1)
                if Phix.dim() > 2:
                    Phix = Phix.view(Phix.shape[0], -1)

            batch_x = torch.stack(kwargs["gt"])
            batch_x = batch_x.squeeze(dim=1).to(device)
            ann_paths = kwargs["ann_path"]
            file_id = kwargs["file_id"]

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        layers_sym = []
        intermediate = [x]

        for i in range(self.LayerNo):
            x, layer_sym = self.fcs[i](x, PhiTPhi, PhiTb)
            intermediate.append(x)
            layers_sym.append(layer_sym)

        x_final = x
        subpixel_offset = None
        if self.use_subpixel_offset:
            x_final_map = x_final.view(-1, 1, 11 * self.c, 11 * self.c)
            if mode == 'loss' and self.subpixel_offset_detach:
                offset_input = x_final_map.detach()
            else:
                offset_input = x_final_map
            subpixel_offset = self.subpixel_offset_head(offset_input)

        if mode == 'tensor':
            return x_final
        elif mode == 'predict':
            targets_GT = read_targets_from_xml_list(ann_paths, self.is_mat)
            output = {
                "x_final": x_final,
                "targets_GT": targets_GT,
                "file_id": file_id,
                "Input_image": Input_image,
                "img_gt": batch_x,
            }
            if subpixel_offset is not None:
                output["subpixel_offset"] = subpixel_offset
            return [output]
        elif mode == 'loss':
            loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))
            loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
            for k in range(self.LayerNo - 1):
                loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
            gamma = torch.Tensor([0.01]).to(device)

            losses = {
                'loss_discrepancy': loss_discrepancy,
                'loss_constraint': torch.mul(gamma, loss_constraint),
            }
            if subpixel_offset is not None:
                offset_target, offset_mask = self.build_subpixel_offset_targets(
                    ann_paths,
                    height=11 * self.c,
                    width=11 * self.c)
                if torch.any(offset_mask):
                    valid_mask = offset_mask.to(subpixel_offset.dtype)
                    diff = torch.abs(subpixel_offset - offset_target)
                    diff = diff * valid_mask
                    beta = self.subpixel_offset_beta
                    loss = torch.where(
                        diff < beta,
                        0.5 * diff * diff / beta,
                        diff - 0.5 * beta)
                    loss_subpixel_offset = (
                        loss.sum() / (valid_mask.sum() * 2.0))
                    losses['loss_subpixel_offset'] = (
                        self.subpixel_offset_loss_weight *
                        loss_subpixel_offset)
                else:
                    losses['loss_subpixel_offset'] = subpixel_offset.sum() * 0.0
            return losses
