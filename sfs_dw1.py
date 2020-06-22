"""
Copyright (c) Baidu, Inc. and its affiliates. All Rights Reserved.
"""
import torch.nn.functional as F
from torch import nn
import torch

class Yolo(nn.Module):
    """
        Arguments:
    """

    def __init__(self, cfg):
        super(Yolo, self).__init__()
        self.downx2 = BaseBlockRelu(3, 16, 3, stride=2, padding=1, bias=False)
        self.conv1 = BaseBlockRelu(16, 16, 3, stride=2, padding=1, bias=False)

        self.conv1_input_4 = BaseBlockRelu(16, 32, 1, stride=1, padding=0, bias=False)

        self.conv2_1 = BaseBlockRelu(16, 32, 3, stride=2, padding=1, bias=False)
        self.conv2_2 = BaseBlockRelu(32, 16, 1, stride=1, padding=0, bias=False)
        self.conv2_3 = BaseBlockRelu(16, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_branch2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_branch2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.res2b_branch2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_branch2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.skip_8 = BaseBlockRelu(32, 64, 1, stride=1, padding=0, bias=False)

        self.conv4_1 = BaseBlockRelu(32, 128, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = BaseBlockRelu(128, 64, 1, stride=1, padding=0, bias=False)
        self.conv4_3 = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_branch2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_branch2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4b_branch2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4b_branch2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4c_branch2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4c_branch2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4d_branch2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4d_branch2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.skip_16 = BaseBlockRelu(128, 128, 1, stride=1, padding=0, bias=False)

        self.conv5_1 = BaseBlockRelu(128, 256, 3, stride=2, padding=1, bias=False)
        self.conv5_2 = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.conv5_3 = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.res5a_branch2a = BaseBlockRelu(256, 256, 3, stride=1, padding=1, bias=False)
        self.res5a_branch2b = BaseBlock(256, 256, 3, stride=1, padding=1, bias=False)
        self.conv5_4 = BaseBlockRelu(256, 256, 3, stride=1, padding=1, bias=False)

        self.deconv5 = BaseBlockDeconvRelu(256, 128, 2, stride=2, padding=0, bias=False)
        self.res5d_seg = BaseBlockRelu(128, 128, 1, stride=1, padding=0, bias=False)

        self.deconv4 = BaseBlockDeconvRelu(128, 64, 2, stride=2, padding=0, bias=False)
        self.res4d_seg = BaseBlockRelu(128, 64, 1, stride=1, padding=0, bias=False)

        self.deconv3 = BaseBlockDeconvRelu(64, 32, 2, stride=2, padding=0, bias=False)
        self.res3d_seg = BaseBlockRelu(64, 32, 1, stride=1, padding=0, bias=False)

        self.conv8_1 = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.conv8_2 = BaseBlockRelu(32, 32, 1, stride=1, padding=0, bias=False)

        # layer
        self.layer1 = nn.Conv2d(32, 26, 3, 1, 1)
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.downx2(x)
        x = self.conv1(x)
        x_b1 = self.conv1_input_4(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = x + self.res2a_branch2b(self.res2a_branch2a(x))
        x = F.relu_(x)

        x = x + self.res2b_branch2b(self.res2b_branch2a(x))
        x = F.relu_(x)

        x_b2 = self.skip_8(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = x + self.res4a_branch2b(self.res4a_branch2a(x))
        x = F.relu_(x)

        x = x + self.res4b_branch2b(self.res4b_branch2a(x))
        x = F.relu_(x)

        x = x + self.res4c_branch2b(self.res4c_branch2a(x))
        x = F.relu_(x)

        x = x + self.res4d_branch2b(self.res4d_branch2a(x))
        x = F.relu_(x)

        x = self.skip_16(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = x + self.res5a_branch2b(self.res5a_branch2a(x))
        x = F.relu_(x)
        x = self.conv5_4(x)

        x = self.deconv5(x)
        x = self.res5d_seg(x)

        x = self.deconv4(x)
        x = self.res4d_seg(torch.cat((x, x_b2), 1))

        x = self.deconv3(x)
        x = self.res3d_seg(torch.cat((x, x_b1), 1))

        x = self.conv8_1(x)
        x = self.conv8_2(x)
        x = self.layer1(x)

        return x

class SfsVps(nn.Module):
    """
        Arguments:
    """

    def __init__(self, cfg):
        super(SfsVps, self).__init__()
        self.downx2 = BaseBlockRelu(3, 16, 3, stride=2, padding=1, bias=False)
        self.conv1 = BaseBlockRelu(16, 16, 3, stride=2, padding=1, bias=False)

        self.conv1_input_4 = BaseBlockRelu(16, 32, 1, stride=1, padding=0, bias=False)

        self.conv2_1 = BaseBlockRelu(16, 32, 3, stride=2, padding=1, bias=False)
        self.conv2_2 = BaseBlockRelu(32, 16, 1, stride=1, padding=0, bias=False)
        self.conv2_3 = BaseBlockRelu(16, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_branch2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_branch2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.skip_8 = BaseBlockRelu(32, 64, 1, stride=1, padding=0, bias=False)

        self.conv4_1 = BaseBlockRelu(32, 128, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = BaseBlockRelu(128, 64, 1, stride=1, padding=0, bias=False)
        self.conv4_3 = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_branch2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_branch2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.conv4_4 = BaseBlockRelu(128, 64, 1, stride=1, padding=0, bias=False)
        self.conv4_5 = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)
        self.res4b_branch2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4b_branch2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.conv4_6 = BaseBlockRelu(128, 64, 1, stride=1, padding=0, bias=False)
        self.conv4_7 = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)
        self.res4c_branch2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4c_branch2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.skip_16 = BaseBlockRelu(128, 128, 1, stride=1, padding=0, bias=False)

        self.conv5_1 = BaseBlockRelu(128, 256, 3, stride=2, padding=1, bias=False)
        self.conv5_2 = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.conv5_3 = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.res5a_branch2a = BaseBlockRelu(256, 256, 3, stride=1, padding=1, bias=False)
        self.res5a_branch2b = BaseBlock(256, 256, 3, stride=1, padding=1, bias=False)
        self.conv5_4 = BaseBlockRelu(256, 256, 3, stride=1, padding=1, bias=False)

        self.conv8_1 = BaseBlockRelu(256, 128, 3, stride=1, padding=1, bias=False)
        self.conv8_2 = BaseBlockRelu(128, 64, 1, stride=1, padding=0, bias=False)
        self.conv8_3 = BaseBlockRelu(64, 32, 1, stride=1, padding=0, bias=False)

        # layer
        self.layer1 = nn.Conv2d(32, 26, 3, 1, 1)
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.downx2(x)
        x = self.conv1(x)
        x_b1 = self.conv1_input_4(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = x + self.res2a_branch2b(self.res2a_branch2a(x))
        x = F.relu_(x)

        x_b2 = self.skip_8(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = x + self.res4a_branch2b(self.res4a_branch2a(x))
        x = F.relu_(x)

        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = x + self.res4b_branch2b(self.res4b_branch2a(x))
        x = F.relu_(x)

        x = self.conv4_6(x)
        x = self.conv4_7(x)
        x = x + self.res4c_branch2b(self.res4c_branch2a(x))
        x = F.relu_(x)

        x = self.skip_16(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = x + self.res5a_branch2b(self.res5a_branch2a(x))
        x = F.relu_(x)
        x = self.conv5_4(x)

        x = self.conv8_1(x)
        x = self.conv8_2(x)
        x = self.conv8_3(x)
        x = self.layer1(x)

        return x

class BaseBlock(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(BaseBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBlock(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        return x


class BaseBlockDeconvRelu(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(BaseBlockDeconvRelu, self).__init__()
        self.conv = nn.ConvTranspose2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x


class BaseBlockRelu(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(BaseBlockRelu, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x


class BaseBlockReluPool(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(BaseBlockReluPool, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        x = self.pool(x)
        return x
