import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from ..tricks.attention_block import LSKmodule


class DIST_BasicBlock(torch.nn.Module):
  def __init__(self, **kwargs):
    super(DIST_BasicBlock, self).__init__()

    c = kwargs['c']
    lambda_weight = kwargs['lambda_weight']

    self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

    self.conv1_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 1, 3, 3)))
    self.conv2_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))

    self.conv1_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
    self.conv2_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(1, 64, 3, 3)))

    self.w1 = nn.Linear(11 * 11 * c * c, 256)
    self.w2 = nn.Linear(256, 9)

    self.att = LSKmodule(64)

    self.register_buffer('lambda_weight', torch.tensor([lambda_weight]).float())
    self.c = c

  @staticmethod
  def per_sample_conv2d(x, kernels, padding=1):
    batch_size, _, height, width = x.shape
    grouped_input = x.reshape(1, batch_size, height, width)
    grouped_kernels = kernels.reshape(batch_size, 1, kernels.shape[-2],
                                      kernels.shape[-1])
    output = F.conv2d(
      grouped_input,
      grouped_kernels,
      stride=1,
      padding=padding,
      groups=batch_size)
    return output.reshape(batch_size, 1, output.shape[-2], output.shape[-1])

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
    x = self.per_sample_conv2d(x_input, weights, padding=1)
    x = torch.sigmoid(x)

    x_forward = self.lambda_weight * x_forward + (1 - self.lambda_weight) * x

    x = torch.mul(torch.sign(x_forward),
                  F.relu(torch.abs(x_forward) - self.att(x_forward)))

    x = F.conv2d(x, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_backward = F.conv2d(x, self.conv2_backward, padding=1)

    x_pred = x_backward.view(-1, 11 * self.c * 11 * self.c)

    x = F.conv2d(x_forward, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_est = F.conv2d(x, self.conv2_backward, padding=1)

    symloss = x_est - x_input

    return [x_pred, symloss]
