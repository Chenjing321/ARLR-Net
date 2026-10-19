import torch
import torch.nn as nn
import torch.nn.functional as F


class LSKmodule(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.convl = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
    self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
    self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
    self.conv_squeeze = nn.Conv2d(2, 2, 3, padding=1)
    self.conv_m = nn.Conv2d(dim // 2, dim, 1)

  def forward(self, x):
    attn1 = x
    attn2 = self.convl(attn1)

    attn1 = self.conv0_s(attn1)
    attn2 = self.conv1_s(attn2)

    attn = torch.cat([attn1, attn2], dim=1)
    avg_attn = torch.mean(attn, dim=1, keepdim=True)
    max_attn, _ = torch.max(attn, dim=1, keepdim=True)
    agg = torch.cat([avg_attn, max_attn], dim=1)
    sig = self.conv_squeeze(agg).sigmoid()

    attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
           attn2 * sig[:, 1, :, :].unsqueeze(1)
    attn = self.conv_m(attn)

    return x * attn


class AdaptiveLSKThreshold(nn.Module):
  """LSK threshold with residual local, channel, and token context gates.

  The module starts as the original LSK threshold because every residual gate is
  zero-initialized. During training it can adapt the threshold with channel
  statistics, local energy, and pooled self-attention context.
  """

  def __init__(self,
               dim,
               reduction=8,
               local_hidden=8,
               attn_dim=32,
               num_heads=4,
               pool_size=11,
               residual_scale=0.25):
    super().__init__()
    self.base_threshold = LSKmodule(dim)
    self.residual_scale = residual_scale
    self.pool_size = pool_size

    channel_hidden = max(8, dim // reduction)
    self.channel_gate = nn.Sequential(
        nn.Conv2d(dim * 2, channel_hidden, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel_hidden, dim, kernel_size=1),
    )

    self.local_gate = nn.Sequential(
        nn.Conv2d(3, local_hidden, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(local_hidden, 1, kernel_size=1),
    )

    self.token_proj = nn.Conv2d(dim, attn_dim, kernel_size=1)
    self.token_norm = nn.LayerNorm(attn_dim)
    self.self_attn = nn.MultiheadAttention(
        embed_dim=attn_dim,
        num_heads=num_heads,
        batch_first=True)
    self.context_out = nn.Conv2d(attn_dim, 1, kernel_size=1)

    nn.init.zeros_(self.channel_gate[-1].weight)
    nn.init.zeros_(self.channel_gate[-1].bias)
    nn.init.zeros_(self.local_gate[-1].weight)
    nn.init.zeros_(self.local_gate[-1].bias)
    nn.init.zeros_(self.context_out.weight)
    nn.init.zeros_(self.context_out.bias)

  def forward(self, x):
    base_threshold = self.base_threshold(x)
    abs_x = torch.abs(x)

    channel_avg = F.adaptive_avg_pool2d(abs_x, output_size=1)
    channel_max = F.adaptive_max_pool2d(abs_x, output_size=1)
    channel_delta = self.channel_gate(
        torch.cat([channel_avg, channel_max], dim=1))

    energy = torch.mean(abs_x, dim=1, keepdim=True)
    local_mean = F.avg_pool2d(energy, kernel_size=3, stride=1, padding=1)
    local_max = F.max_pool2d(energy, kernel_size=3, stride=1, padding=1)
    local_delta = self.local_gate(
        torch.cat([energy, local_mean, local_max], dim=1))

    pooled_h = min(self.pool_size, x.shape[-2])
    pooled_w = min(self.pool_size, x.shape[-1])
    tokens = F.adaptive_avg_pool2d(x, output_size=(pooled_h, pooled_w))
    tokens = self.token_proj(tokens)
    batch_size, token_dim, token_h, token_w = tokens.shape
    tokens = tokens.flatten(2).transpose(1, 2)
    tokens = self.token_norm(tokens)
    context, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
    context = context.transpose(1, 2).reshape(
        batch_size, token_dim, token_h, token_w)
    context = F.interpolate(
        context,
        size=x.shape[-2:],
        mode='bilinear',
        align_corners=False)
    context_delta = self.context_out(context)

    delta = channel_delta + local_delta + context_delta
    scale = 1.0 + self.residual_scale * torch.tanh(delta)
    return base_threshold * scale
