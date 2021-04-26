import torch
from torch import nn


class EGNNGlobalPool(nn.Module):

    def __init__(self, dim=None, tensor_dim=1, mean=False):
        super().__init__()
        self.dim = dim
        self.mean = mean
        self.tensor_dim = tensor_dim

    def forward(self, x):
        if len(x) == 2:
            if self.mean:
                if self.dim is None:
                    return x.mean(self.tensor_dim)
                else:
                    return x[self.dim].mean(self.tensor_dim)
            else:
                if self.dim is None:
                    return x.sum(self.tensor_dim)
                else:
                    return x[self.dim].sum(self.tensor_dim)
        vals, coords, mask = x

        if self.mean:
            # mean pooling
            summed = torch.where(mask.unsqueeze(-1), vals,
                                 torch.zeros_like(vals)).sum(1)
            summed_mask = mask.sum(-1).unsqueeze(-1)
            summed_mask = torch.where(
                summed_mask == 0, torch.ones_like(summed_mask), summed_mask)
            summed /= summed_mask

            return [summed, coords, mask]
        else:
            # max pooling
            masked = torch.where(
                mask.unsqueeze(-1),
                vals,
                torch.tensor(
                    -1e38,
                    dtype=vals.dtype,
                    device=vals.device,
                )
                * torch.ones_like(vals),
            )

            return [masked.max(dim=1)[0], coords, mask]


class EGNNBatchNorm(nn.BatchNorm1d):

    def forward(self, inp):
        if len(inp) == 2:
            x, coords = inp
            mask = torch.ones(coords.shape[:-1]).byte().cuda()
        else:
            x, coords, mask = inp
        sum_dims = list(range(len(x.shape[:-1])))
        x_or_zero = torch.where(mask.unsqueeze(-1), x,
                                torch.zeros_like(x))  # remove nans
        if self.training or not self.track_running_stats:
            xsum = x_or_zero.sum(dim=sum_dims)
            xxsum = (x_or_zero * x_or_zero).sum(dim=sum_dims)
            numel_notnan = (mask).sum()
            xmean = xsum / numel_notnan
            sumvar = xxsum - xsum * xmean
            unbias_var = sumvar / (numel_notnan - 1)
            bias_var = sumvar / numel_notnan
            self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * xmean.detach())
            self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * unbias_var.detach())
        else:
            xmean, bias_var = self.running_mean, self.running_var
        std = bias_var.clamp(self.eps) ** 0.5
        ratio = self.weight / std
        output = (x_or_zero * ratio + (self.bias - xmean * ratio))
        return (output, coords)
