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
            in_channels：    输入的通道数。
            out_channels：   输出的通道数。
            kernel_size:    卷积核的大小。默认使用 2×2 的卷积核
            stride：        卷积核移动步长。默认为 1
            padding：       填充。默认无填充
            bias：          卷积后的偏置。默认添加偏置

        示例：
            contracting_block_1 = ContractingBlock(3, 64)
            contracting_block_2 = ContractingBlock(3, 64, 3, 1, 1, True)
        """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=1, bias=True):
        super(UpSamplingBlock, self).__init__()
        self.tran_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x, concat_features):
        tran_conv_out = self.tran_conv(x)
        out = tran_conv_out
        for concat_feature in concat_features:
            out = torch.cat((concat_feature, out), dim=1)
        return out


class UNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = DoubleConvBlock(1, 64)

        self.down_sampling_1 = nn.MaxPool2d(2, 2)
        self.conv_block_2 = DoubleConvBlock(64, 128)
        self.up_sampling_1 = UpSamplingBlock(128, 64)
        self.conv_block_3 = DoubleConvBlock(128, 64)

        self.down_sampling_2 = nn.MaxPool2d(2, 2)
        self.conv_block_4 = DoubleConvBlock(128, 256)
        self.up_sampling_2 = UpSamplingBlock(256, 128)
        self.conv_block_5 = DoubleConvBlock(256, 128)
        self.up_sampling_3 = UpSamplingBlock(128, 64)
        self.conv_block_6 = DoubleConvBlock(192, 64)

        self.down_sampling_3 = nn.MaxPool2d(2, 2)
        self.conv_block_7 = DoubleConvBlock(256, 512)
        self.up_sampling_4 = UpSamplingBlock(512, 256)
        self.conv_block_8 = DoubleConvBlock(512, 256)
        self.up_sampling_5 = UpSamplingBlock(256, 128)
        self.conv_block_9 = DoubleConvBlock(384, 128)
        self.up_sampling_6 = UpSamplingBlock(128, 64)
        self.conv_block_10 = DoubleConvBlock(256, 64)

        self.down_sampling_4 = nn.MaxPool2d(2, 2)
        self.conv_block_11 = DoubleConvBlock(512, 1024)
        self.up_sampling_7 = UpSamplingBlock(1024, 512)
        self.conv_block_12 = DoubleConvBlock(1024, 512)
        self.up_sampling_8 = UpSamplingBlock(512, 256)
        self.conv_block_13 = DoubleConvBlock(768, 256)
        self.up_sampling_9 = UpSamplingBlock(256, 128)
        self.conv_block_14 = DoubleConvBlock(512, 128)
        self.up_sampling_10 = UpSamplingBlock(128, 64)
        self.conv_block_15 = DoubleConvBlock(320, 64)

        self.out = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        node_1_1 = self.conv_block_1(x)

        node_2_1 = self.conv_block_2(self.down_sampling_1(node_1_1))
        node_1_2 = self.conv_block_3(self.up_sampling_1(node_2_1, [node_1_1]))

        node_3_1 = self.conv_block_4(self.down_sampling_2(node_2_1))
        node_2_2 = self.conv_block_5(self.up_sampling_2(node_3_1, [node_2_1]))
        node_1_3 = self.conv_block_6(self.up_sampling_3(node_2_2, [node_1_1, node_1_2]))

        node_4_1 = self.conv_block_7(self.down_sampling_3(node_3_1))
        node_3_2 = self.conv_block_8(self.up_sampling_4(node_4_1, [node_3_1]))
        node_2_3 = self.conv_block_9(self.up_sampling_5(node_3_2, [node_2_1, node_2_2]))
        node_1_4 = self.conv_block_10(self.up_sampling_6(node_2_3, [node_1_1, node_1_2, node_1_3]))

        node_5_1 = self.conv_block_11(self.down_sampling_4(node_4_1))
        node_4_2 = self.conv_block_12(self.up_sampling_7(node_5_1, [node_4_1]))
        node_3_3 = self.conv_block_13(self.up_sampling_8(node_4_2, [node_3_1, node_3_2]))
        node_2_4 = self.conv_block_14(self.up_sampling_9(node_3_3, [node_2_1, node_2_2, node_2_3]))
        node_1_5 = self.conv_block_15(self.up_sampling_10(node_2_4, [node_1_1, node_1_2, node_1_3, node_1_4]))

        out = self.out(node_1_5)

        return out
