import torch
import torch.nn.functional as F
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    """UNet 两层卷积块
        收缩模块经过了两次卷积操作，每一次卷积之后都进行一次 relu 操作
        参数：
            in_channels：    输入的通道数。
            out_channels：   输出的通道数。
            kernel_size:    卷积核的大小。默认使用 3×3 的卷积核
            stride：        卷积核移动步长。默认为 1
            padding：       填充。默认无填充
            bias：          卷积后的偏置。默认添加偏置

        示例：
            contracting_block_1 = ContractingBlock(3, 64)
            contracting_block_2 = ContractingBlock(3, 64, 3, 1, 1, True)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(DoubleConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv_out_1 = self.conv_1(x)
        bn_out_1 = self.bn_1(conv_out_1)
        relu_out = F.relu(bn_out_1)
        conv_out_2 = self.conv_2(relu_out)
        bn_out_2 = self.bn_2(conv_out_2)
        return F.relu(bn_out_2)


class UpSamplingBlock(nn.Module):
    """UNet 上采样和拼接模块
        收缩模块经过了两次卷积操作，每一次卷积之后都进行一次 relu 操作
        参数：
            in_channels：    转置卷积输入的通道数。
            out_channels：   转置卷积输出的通道数。
            kernel_size:    转置卷积的卷积核的大小。默认使用 2×2 的卷积核
            stride：        转置卷积的卷积核移动步长。默认为 1
            padding：       填充。默认无填充
            bias：          卷积后的偏置。默认添加偏置

        示例：
            contracting_block_1 = ContractingBlock(3, 64)
            contracting_block_2 = ContractingBlock(3, 64, 3, 1, 1, True)
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True):
        super(UpSamplingBlock, self).__init__()
        self.tran_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x, concat_feature):
        tran_conv_out = self.tran_conv(x)
        return torch.cat((concat_feature, tran_conv_out), dim=1)


class UNet(nn.Module):
    '''UNet 网络架构
    lr 0.001
    '''


    def __init__(self):
        super().__init__()

        self.conv_block_1 = DoubleConvBlock(1, 32)
        self.down_sampling_1 = nn.MaxPool2d(2, 2)
        self.conv_block_2 = DoubleConvBlock(32, 64)
        self.down_sampling_2 = nn.MaxPool2d(2, 2)
        self.conv_block_3 = DoubleConvBlock(64, 128)
        self.down_sampling_3 = nn.MaxPool2d(2, 2)
        self.conv_block_4 = DoubleConvBlock(128, 256)
        self.down_sampling_4 = nn.MaxPool2d(2, 2)
        self.conv_block_5 = DoubleConvBlock(256, 512)
        self.up_sampling_1 = UpSamplingBlock(512, 256)
        self.conv_block_6 = DoubleConvBlock(512, 256)
        self.up_sampling_2 = UpSamplingBlock(256, 128)
        self.conv_block_7 = DoubleConvBlock(256, 128)
        self.up_sampling_3 = UpSamplingBlock(128, 64)
        self.conv_block_8 = DoubleConvBlock(128, 64)
        self.up_sampling_4 = UpSamplingBlock(64, 32)
        self.conv_block_9 = DoubleConvBlock(64, 32)
        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        conv_block_out_1 = self.conv_block_1(x)
        conv_block_out_2 = self.conv_block_2(self.down_sampling_1(conv_block_out_1))
        conv_block_out_3 = self.conv_block_3(self.down_sampling_2(conv_block_out_2))
        conv_block_out_4 = self.conv_block_4(self.down_sampling_3(conv_block_out_3))
        conv_block_out_5 = self.conv_block_5(self.down_sampling_4(conv_block_out_4))

        conv_block_out_6 = self.conv_block_6(self.up_sampling_1(conv_block_out_5, conv_block_out_4))
        conv_block_out_7 = self.conv_block_7(self.up_sampling_2(conv_block_out_6, conv_block_out_3))
        conv_block_out_8 = self.conv_block_8(self.up_sampling_3(conv_block_out_7, conv_block_out_2))
        conv_block_out_9 = self.conv_block_9(self.up_sampling_4(conv_block_out_8, conv_block_out_1))

        out = self.out(conv_block_out_9)

        return out
