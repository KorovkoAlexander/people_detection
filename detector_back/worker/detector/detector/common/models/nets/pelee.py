import functools
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_planes, out_planes, bias=False, **kwargs),
                        nn.BatchNorm2d(out_planes),
                        nn.ReLU(inplace=True),
                    )

    def forward(self, x):
        return self.conv(x)


class StemBlock(nn.Module):
    """
    StemBlock used in PeleeNet
    According to Pelee paper, it is motivated by
    Inception-v4 Szegedy et al. (2017) and DSOD Shen et al. (2017)
    This is used before the first dense layer
    """

    def __init__(self, k=32):
        super(StemBlock, self).__init__()
        self.conv1 = ConvBlock(3, k, kernel_size=3, stride=2, padding=1)
        self.left_conv1 = ConvBlock(k, k//2, kernel_size=1, stride=1)
        self.left_conv2 = ConvBlock(k//2, k, kernel_size=3, stride=2, padding=1)
        self.right = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_last = ConvBlock(k*2, k, kernel_size=1, stride=1)

    def forward(self, x):
        """
        x: input image of shape [batch, 3, 224, 224]
        """
        x = self.conv1(x)                    # [batch, 32, 112, 112]
        left = self.left_conv1(x)            # [batch, 16, 112, 112]
        left = self.left_conv2(left)         # [batch, 32, 112, 112]
        right = self.right(x)                # [batch, 32, 112, 112]
        x = torch.cat((left, right), dim=1)  # [batch, 64, 112, 112]
        x = self.conv_last(x)                # [batch, 32,  56,  56]
        return x


class DenseLayer(nn.Module):
    """
    Two-way dense layer suggested by the paper
    """
    def __init__(self, in_planes, growth_rate, bottleneck_width):
        """
        bottleneck_width is usally 1, 2, or 4
        """
        super(DenseLayer, self).__init__()

        inter_channel = bottleneck_width * growth_rate / 2
        inter_channel = bottleneck_width * growth_rate // 2  # will be k/2, k, 2k depending on bottleneck_width = 1,2,4

        # Left side
        self.cb1_a = ConvBlock(in_planes, inter_channel, kernel_size=1, stride=1)
        self.cb1_b = ConvBlock(inter_channel, growth_rate//2, kernel_size=3, stride=1, padding=1)


        # Right side
        self.cb2_a = ConvBlock(in_planes, inter_channel, kernel_size=1, stride=1)
        self.cb2_b = ConvBlock(inter_channel, growth_rate//2, kernel_size=3, stride=1, padding=1)
        self.cb2_c = ConvBlock(growth_rate//2, growth_rate//2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = torch.cat((x, cb1_b_out, cb2_c_out), 1)

        return out


class DenseBlock(nn.Module):
    def __init__(self, in_planes, no_dense_layers, growth_rate, bottleneck_width):
        super(DenseBlock, self).__init__()
        layers = [DenseLayer(in_planes+growth_rate*i, growth_rate, bottleneck_width) for i in range(no_dense_layers)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, inp, oup, last=False):
        super(TransitionLayer, self).__init__()
        conv = ConvBlock(inp, oup, kernel_size=1, stride=1)
        if not last:
            self.layer = nn.Sequential(conv, nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.layer = conv

    def forward(self, x):
        return self.layer(x)


def _pelee(growth_rate, dense_layers, bottleneck_widths, num_layers):
    layers = [StemBlock(k=64)]
    filters = 64
    for i in range(num_layers):
        next_filters = filters + growth_rate * dense_layers[i]
        layers.append(
            nn.Sequential(
                DenseBlock(filters, dense_layers[i], growth_rate, bottleneck_widths[i]),
                TransitionLayer(next_filters, next_filters, last=False)
                #TransitionLayer(next_filters, next_filters, last=(i == num_layers-1))
            )
        )
        filters += growth_rate * dense_layers[i]
    return layers


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


pelee = wrapped_partial(
    _pelee,
    growth_rate=48,
    dense_layers=[3, 4, 8, 6],
    bottleneck_widths=[1, 2, 4, 4],
    num_layers=4
)
