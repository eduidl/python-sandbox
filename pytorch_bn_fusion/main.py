from typing import Tuple
import unittest

from parameterized import parameterized  # type: ignore
import torch
from torch import nn


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    A, bias = convert_bn_params(bn)
    conv.weight.data.mul_(A.transpose(0, 1))
    conv.bias = nn.Parameter(bias.squeeze())

    return conv


def convert_bn_params(bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    var_rsqrt = (bn.running_var + bn.eps).rsqrt()
    A = bn.weight * var_rsqrt
    bias = bn.bias - A * bn.running_mean
    return A.view(1, -1, 1, 1), bias.view(1, -1, 1, 1)


def gen_conv(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)


def gen_bn(num_features: int) -> nn.BatchNorm2d:
    bn = nn.BatchNorm2d(num_features)
    bn.weight = nn.Parameter(torch.rand(num_features))
    bn.bias = nn.Parameter(torch.rand(num_features))
    bn.running_mean = torch.rand(num_features)
    bn.running_var = torch.rand(num_features)

    return bn.eval()


class TestBatchNormFusion(unittest.TestCase):

    BATCH_SIZE: int = 128
    WIDTH: int = 128
    HEIGHT: int = 128

    @parameterized.expand([(10,), (20,), (30,)])
    def test_fuse_bn(self, num_features: int) -> None:
        bn = gen_bn(num_features)

        input = torch.randn(self.BATCH_SIZE, num_features, self.HEIGHT, self.WIDTH)
        out1 = bn(input)

        A, bias = convert_bn_params(bn)
        out2 = input * A + bias

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    @parameterized.expand([(10, 15), (20, 20), (30, 25)])
    def test_fuse_conv_bn(self, in_channels: int, out_channels: int) -> None:
        conv = gen_conv(in_channels, out_channels)
        bn = gen_bn(out_channels)

        input = torch.randn(self.BATCH_SIZE, in_channels, self.HEIGHT, self.WIDTH)
        out1 = bn(conv(input))

        fused_conv = fuse_conv_bn(conv, bn)
        out2 = fused_conv(input)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
