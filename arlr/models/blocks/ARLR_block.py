import torch
from torch.nn import init
import torch.nn.functional as F
from ..tricks.attention_block import *
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ARLR_BasicBlock(torch.nn.Module):
  def __init__(self, **kwargs):
    super(ARLR_BasicBlock, self).__init__()

    c = kwargs['c']
    lambda_weight = kwargs['lambda_weight']
    threshold_module = kwargs.get('threshold_module', 'AdaptiveLSKThreshold')
    self.dynamic_conv_module = kwargs.get(
      'dynamic_conv_module', 'CorrectedDynamicKernelMixture')

    self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

    self.conv1_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 1, 3, 3)))
    self.conv2_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))

    self.conv1_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
    self.conv2_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(1, 64, 3, 3)))

    self.w1 = nn.Linear(11*11*c*c, 256)
    self.w2 = nn.Linear(256, 9)
    if self.dynamic_conv_module not in [
        'OriginalDynamicConv', 'CorrectedOriginalDynamicConv',
        'AdaptiveDynamicConv', 'DynamicKernelMixture',
        'CorrectedDynamicKernelMixture']:
      raise KeyError(f'Unknown dynamic conv module: {self.dynamic_conv_module}')
    #消融
    if self.dynamic_conv_module == 'AdaptiveDynamicConv':
      self.dynamic_refine_w2 = nn.Linear(256, 9)
      self.dynamic_refine_scale = nn.Parameter(torch.Tensor([0.1]))
      nn.init.zeros_(self.dynamic_refine_w2.weight)
      nn.init.zeros_(self.dynamic_refine_w2.bias)
    #CorrectedDynamicKernelMixture
    if self.dynamic_conv_module in [
        'DynamicKernelMixture', 'CorrectedDynamicKernelMixture']:
      self.kernel_mixture_num = kwargs.get('dynamic_kernel_mixture_num', 4)
      self.kernel_mixture_w2 = nn.Linear(256, self.kernel_mixture_num * 9)
      self.kernel_mixture_gate = nn.Linear(256, self.kernel_mixture_num)
      self.kernel_mixture_scale = nn.Parameter(torch.Tensor([0.1]))
      nn.init.zeros_(self.kernel_mixture_w2.weight)
      nn.init.zeros_(self.kernel_mixture_w2.bias)
      nn.init.zeros_(self.kernel_mixture_gate.weight)
      nn.init.zeros_(self.kernel_mixture_gate.bias)
    if self.dynamic_conv_module in [
        'CorrectedOriginalDynamicConv',
        'CorrectedDynamicKernelMixture']:
      self.dynamic_channel_projection = nn.Conv2d(1, 64, kernel_size=1)
      nn.init.ones_(self.dynamic_channel_projection.weight)
      nn.init.zeros_(self.dynamic_channel_projection.bias)

    module_dict = globals()
    if threshold_module not in module_dict:
      raise KeyError(f'Unknown threshold module: {threshold_module}')
    self.att = module_dict[threshold_module](64)

    self.lambda_weight = torch.Tensor([lambda_weight]).to(device)
    self.c = c

  @staticmethod
  def per_sample_conv2d(x, kernels, padding=1):
    """Apply one or more single-input kernels to each sample independently."""
    batch_size, _, height, width = x.shape
    output_channels = kernels.shape[1]
    grouped_input = x.reshape(1, batch_size, height, width)
    grouped_kernels = kernels.reshape(
      batch_size * output_channels, 1, kernels.shape[-2], kernels.shape[-1])
    output = F.conv2d(
      grouped_input,
      grouped_kernels,
      stride=1,
      padding=padding,
      groups=batch_size)
    return output.reshape(
      batch_size, output_channels, output.shape[-2], output.shape[-1])

  def forward(self, x, PhiTPhi, PhiTb):
    minor = x

    x = x - self.lambda_step * torch.mm(x, PhiTPhi)
    x = x + self.lambda_step * PhiTb
    x_input = x.view(-1, 1, 11 * self.c, 11 * self.c)

    x = F.conv2d(x_input, self.conv1_forward, padding=1)
    x = F.relu(x)
    x_forward = F.conv2d(x, self.conv2_forward, padding=1)

    minor = self.w1(minor)
    weights = self.w2(minor)

    weights = weights.reshape(-1, 1, 3, 3)
    if self.dynamic_conv_module == 'CorrectedDynamicKernelMixture':
      mixture_feat = F.relu(minor)
      residual_weights = self.kernel_mixture_w2(mixture_feat)
      residual_weights = residual_weights.reshape(
        -1, self.kernel_mixture_num, 1, 3, 3)
      mixture_weights = F.softmax(
        self.kernel_mixture_gate(mixture_feat), dim=1)
      mixture_weights = mixture_weights.reshape(
        -1, self.kernel_mixture_num, 1, 1, 1)
      residual_weights = torch.sum(
        residual_weights * mixture_weights, dim=1)
      effective_weights = (
        weights + self.kernel_mixture_scale * residual_weights)
      x = self.per_sample_conv2d(x_input, effective_weights, padding=1)
      x = self.dynamic_channel_projection(x)
    elif self.dynamic_conv_module == 'CorrectedOriginalDynamicConv':
      x = self.per_sample_conv2d(x_input, weights, padding=1)
      x = self.dynamic_channel_projection(x)
    else:
      weights = nn.Parameter(data=weights, requires_grad=False)
      x = F.conv2d(
        input=x_input,
        weight=weights,
        stride=1,
        padding=1,
        groups=1)
      #消融
      if self.dynamic_conv_module == 'AdaptiveDynamicConv':
        refine_weights = self.dynamic_refine_w2(F.relu(minor))
        refine_weights = refine_weights.reshape(-1, 1, 3, 3)
        x_refine = F.conv2d(
          input=x_input,
          weight=refine_weights,
          stride=1,
          padding=2,
          dilation=2,
          groups=1)
        x = x + self.dynamic_refine_scale * x_refine
      #消融
      if self.dynamic_conv_module == 'DynamicKernelMixture':
        mixture_feat = F.relu(minor)
        residual_weights = self.kernel_mixture_w2(mixture_feat)
        residual_weights = residual_weights.reshape(
          -1, self.kernel_mixture_num, 1, 3, 3)
        mixture_weights = F.softmax(
          self.kernel_mixture_gate(mixture_feat), dim=1)
        mixture_weights = mixture_weights.reshape(
          -1, self.kernel_mixture_num, 1, 1, 1)
        residual_weights = torch.sum(
          residual_weights * mixture_weights, dim=1)
        x_residual = F.conv2d(
          input=x_input,
          weight=residual_weights,
          stride=1,
          padding=1,
          groups=1)
        x = x + self.kernel_mixture_scale * x_residual
    x = torch.sigmoid(x)

    x_forward = self.lambda_weight * x_forward + (1 - self.lambda_weight) * x

    threshold = self.att(x_forward)
    x = torch.sign(x_forward) * F.relu(torch.abs(x_forward) - threshold)

    x = F.conv2d(x, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_backward = F.conv2d(x, self.conv2_backward, padding=1)

    x_pred = x_backward.view(-1, 11 * self.c * 11 * self.c)

    x = F.conv2d(x_forward, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_est = F.conv2d(x, self.conv2_backward, padding=1)

    symloss = x_est - x_input

    return [x_pred, symloss]
