import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(x, scale: float =None, size: tuple =None):
    if scale:
        return F.interpolate(x, scale_factor=scale, mode='bilinear')
    if size:
        return F.interpolate(x, size=size, mode="bilinear")
    raise ValueError("Set size or scale!")


def upsample_add(x,y):
    _,_,h,w = y.size()
    return upsample(x, size=(h,w)) + y


class ConvBlock(nn.Module):
    def __init__(self, input, output, kernel, stride, padding=0, use_batchnorm=True):
        super(ConvBlock, self).__init__()
        if use_batchnorm:
            self.seq = nn.Sequential(
                nn.Conv2d(input, output, kernel, stride, padding, bias=False),
                nn.BatchNorm2d(output),
                nn.ReLU(inplace=True),
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv2d(input, output, kernel, stride, padding),
                nn.ReLU(inplace=True),
            )

    def forward(self, input):
        return self.seq(input)


class FFMv1(nn.Module):
    def __init__(self, inp_cnl1, inp_cnl2):
        super(FFMv1, self).__init__()
        self.conv1 = ConvBlock(inp_cnl1, 256, 3, 1, 1)
        self.conv2 = ConvBlock(inp_cnl2, 512, 1, 1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        _,_,h,w = x1.size()
        x2 = upsample(x2, size=(h,w))
        return torch.cat([x1, x2], dim=1)


class FFMv2(nn.Module):
    def __init__(self):
        super(FFMv2, self).__init__()
        self.conv = ConvBlock(768, 128, 1, 1)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)


class TUM(nn.Module):
    def __init__(self, planes=128, scales=6):
        super(TUM, self).__init__()
        self.scales = scales
        self.convs1 = nn.ModuleList(
            [ConvBlock(256, 256, 3, 2, 1) for _ in range(scales-2)] + [ConvBlock(256, 256, 3, 2, 0, use_batchnorm=False)]
        )
        self.convs2 = nn.ModuleList(
            [ConvBlock(256, 256, 3, 1, 1) for _ in range(scales-2)] + [ConvBlock(256, 256, 3, 1, 1, use_batchnorm=False)]
        )
        self.convs3 = nn.ModuleList(
            [ConvBlock(256, planes, 1, 1) for i in range(scales-1)] + [ConvBlock(256, planes, 1, 1, use_batchnorm=False)]
        )

    def forward(self, x):
        outputs1 = [x]
        for i in range(self.scales-1):
            y = outputs1[-1]
            outputs1.append(
                self.convs1[i](y)
            )

        outputs2 = [outputs1[self.scales - 1]]
        for i in range(self.scales-2, -1, -1):
            outputs2.insert(
                0,
                upsample_add(
                    self.convs2[i](outputs2[0]), outputs1[i]
                )
            )
        return [self.convs3[i](outputs2[i]) for i in range(self.scales)]


class SFAM(nn.Module):
    def __init__(self, planes: int=128, num_levels: int=3, scales: int =6, reduce_ratio: int = 16):
        super(SFAM, self).__init__()
        self.scales = scales
        self.fc1 = nn.ModuleList(
            [
                nn.Conv2d(planes * num_levels,
                          planes * num_levels // reduce_ratio,
                          1, 1, 0)
            for _ in range(scales)
            ]
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList(
            [
                nn.Conv2d(
                    planes * num_levels // reduce_ratio,
                    planes * num_levels,
                    1, 1, 0)
                for _ in range(scales)
            ]
        )
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, FP_list):
        tensors = [torch.cat([x[i] for x in FP_list], dim=1) for i in range(self.scales)]
        attention_features = []
        for i , x in enumerate(tensors):
            y = self.avgpool(x)
            y = self.fc1[i](y)
            y = self.relu(y)
            y = self.fc2[i](y)
            y = self.sigmoid(y)
            attention_features.append(x * y)
        return attention_features


class MLFPN(nn.Module):
    def __init__(self, inp_depth1, inp_depth2, num_levels=8, scales=6):
        super(MLFPN, self).__init__()
        self.FP_size = num_levels
        self.ffmv1 = FFMv1(inp_depth1, inp_depth2)
        self.ffmv2s = nn.ModuleList([FFMv2() for _ in range(num_levels)])
        self.tums = nn.ModuleList([TUM(scales=scales) for _ in range(num_levels)])
        self.sfam = SFAM(num_levels = num_levels, scales=scales)
        self.norm = nn.BatchNorm2d(128*num_levels)
        self.conv = ConvBlock(768, 256, 1, 1)

    def forward(self, x1, x2):
        base_feature = self.ffmv1(x1, x2)
        outs = []
        inputs = [self.conv(base_feature)]
        for i in range(self.FP_size - 1):
            x = self.tums[i](inputs[-1])
            outs.append(x)
            inputs.append(
                self.ffmv2s[i](base_feature, x[0])
            )
        outs.append(
            self.tums[-1](inputs[-1])
        )
        sources = self.sfam(outs)
        #sources[0] = self.norm(sources[0])
        return sources


class M2Det(nn.Module):
    def __init__(self, base, head, feature_layer, num_classes, fp_size):
        super(M2Det, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        #self.norm = L2Norm(feature_layer[1][0], 20)

        self.locs = nn.ModuleList(head[0])
        self.confs = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        self.feature_layer = feature_layer[0]
        assert len(feature_layer[1]) == 2, "MLFPN has only 2 inputs"
        self.mlfpn = MLFPN(*feature_layer[1], num_levels=fp_size)

    def forward(self, x, phase='eval'):
        sources, loc, conf = [list() for _ in range(3)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)

        # sources should be sorted by channels ascending
        outputs = self.mlfpn(*sources)

        if phase == 'feature':
            return outputs

        for (x, l, c) in zip(outputs, self.locs, self.confs):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output


def add_extras(base, mbox, num_classes, fp_size):
    loc_layers = []
    conf_layers = []
    in_channels = 128 * fp_size
    for box in mbox:
        loc_layers.append(nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1))
        conf_layers.append(nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1))
    return base, (loc_layers, conf_layers)


def build_ssd(base, feature_layer, mbox, num_classes, fp_size=8):
    base_, head_ = add_extras(base(), mbox, num_classes, fp_size)
    return M2Det(base_, head_, feature_layer, num_classes, fp_size)
