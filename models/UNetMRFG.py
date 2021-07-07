import torch
import torch.nn.functional as F
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1, bias=True)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return


class ResBlock(nn.Module):
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
            contracting_block_1 = ContractingBlock(3, 32)
            contracting_block_2 = ContractingBlock(3, 32, 3, 1, 1, True)
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


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
            contracting_block_1 = ContractingBlock(3, 32)
            contracting_block_2 = ContractingBlock(3, 32, 3, 1, 1, True)
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=1, bias=True):
        super(UpSamplingBlock, self).__init__()
        self.tran_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x, concat_feature):
        tran_conv_out = self.tran_conv(x)

        out = torch.cat((concat_feature, tran_conv_out), dim=1)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        out_channels = 1 if 0 == (in_planes // 16) else in_planes // 16

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, out_channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(out_channels, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttentionBasicBlock(nn.Module):

    def __init__(self, channels):
        super(ChannelAttentionBasicBlock, self).__init__()

        self.ca = ChannelAttention(channels)

    def forward(self, x):
        out = x

        out = self.ca(out) * out

        return out


class FusionBasicBlock(nn.Module):

    def __init__(self, conv_num, tran_conv_num, depth, target_channel, in_channel_list=None, cbam_block=True):
        super(FusionBasicBlock, self).__init__()

        self.cbam_block = cbam_block

        if in_channel_list is None:
            in_channel_list = [1, 32, 64, 128, 256]

        self.layers = nn.ModuleList()

        for i in range(conv_num):
            sequential = nn.Sequential(
                nn.Conv2d(in_channel_list[i], target_channel, 3, padding=1),
                nn.MaxPool2d(2 ** (depth - i - 1), 2 ** (depth - i - 1)),

            )

            if cbam_block:
                sequential.add_module('attention', ChannelAttentionBasicBlock(target_channel))

            self.layers.append(sequential)

        for i in range(tran_conv_num):
            sequential = nn.Sequential(
                nn.ConvTranspose2d(in_channel_list[depth + i], target_channel, 2 ** (i + 1), stride=2 ** (i + 1)),
            )

            if cbam_block:
                sequential.add_module('attention', ChannelAttentionBasicBlock(target_channel))

            self.layers.append(sequential)

        if cbam_block:
            self.sa = SpatialAttention()

    def forward(self, node_list, current_node=None):
        out = current_node

        for index, item in enumerate(self.layers):
            if out is None:
                out = item(node_list[index])
            else:
                out = out + item(node_list[index])

        if self.cbam_block:
            out = self.sa(out) * out

        return out


class MultiResolutionBlock(nn.Module):
    def __init__(self):
        super(MultiResolutionBlock, self).__init__()

        # layer 1(node 1)
        self.fb_basic_block_1_1 = FusionBasicBlock(0, 0, 1, 1)

        # layer 2(生成节点2)
        self.fb_basic_block_2_1 = FusionBasicBlock(1, 0, 2, 32)  # 0 down   1 up

        # layer 3(节点1更新，节点2更新，生成节点3)
        # 节点 1 更新（节点2上采样+节点1）
        self.fb_basic_block_3_1 = FusionBasicBlock(0, 1, 1, 1)
        # 节点 2 更新（节点1下采样+节点2）
        self.fb_basic_block_3_2 = FusionBasicBlock(1, 0, 2, 32)
        # 生成节点 3（节点1下采样+节点2下采样）
        self.fb_basic_block_3_3 = FusionBasicBlock(2, 0, 3, 64)

        # layer 4(节点1更新，节点2更新，节点3更新，生成节点4)
        # 节点 1 更新（节点2上采样+节点3上采样 + 节点1）
        self.fb_basic_block_4_1 = FusionBasicBlock(0, 2, 1, 1)
        # 节点 2 更新（节点1下采样+节点2+节点3上采样）
        self.fb_basic_block_4_2 = FusionBasicBlock(1, 1, 2, 32)
        # 生成节点 3（节点1下采样+节点2下采样+节点3）
        self.fb_basic_block_4_3 = FusionBasicBlock(2, 0, 3, 64)
        # 生成节点 4（节点1下采样+节点2下采样+节点3下采样）
        self.fb_basic_block_4_4 = FusionBasicBlock(3, 0, 4, 128)

        # layer 5(节点1更新，节点2更新，节点3更新，节点4更新，生成节点5)
        # 节点 1 更新（节点2上采样+节点3上采样+节点4上采样 + 节点1）
        self.fb_basic_block_5_1 = FusionBasicBlock(0, 3, 1, 1)
        # 节点 2 更新（节点1下采样+节点2+节点3上采样+节点4上采样）
        self.fb_basic_block_5_2 = FusionBasicBlock(1, 2, 2, 32)
        # 节点 3 更新（节点1下采样+节点2下采样+节点3+节点4上采样）
        self.fb_basic_block_5_3 = FusionBasicBlock(2, 1, 3, 64)
        # 节点 4 更新（节点1下采样+节点2下采样+节点3下采样+节点4）
        self.fb_basic_block_5_4 = FusionBasicBlock(3, 0, 4, 128)
        # 生成节点 5（节点1下采样+节点2下采样+节点3下采样+节点4下采样）
        self.fb_basic_block_5_5 = FusionBasicBlock(4, 0, 5, 256)

    def forward(self, x):
        # layer 1
        input_rel_1 = self.fb_basic_block_1_1([x], current_node=x)

        # layer 2
        input_rel_2 = self.fb_basic_block_2_1([input_rel_1])

        # layer 3
        input_rel_1 = self.fb_basic_block_3_1([input_rel_2], input_rel_1)
        input_rel_2 = self.fb_basic_block_3_2([input_rel_1], input_rel_2)
        input_rel_3 = self.fb_basic_block_3_3([input_rel_1, input_rel_2])

        # layer 4
        input_rel_1 = self.fb_basic_block_4_1([input_rel_2, input_rel_3], input_rel_1)
        input_rel_2 = self.fb_basic_block_4_2([input_rel_1, input_rel_3], input_rel_2)
        input_rel_3 = self.fb_basic_block_4_3([input_rel_1, input_rel_2], input_rel_3)
        input_rel_4 = self.fb_basic_block_4_4([input_rel_1, input_rel_2, input_rel_3])

        # layer 5
        input_rel_1 = self.fb_basic_block_5_1([input_rel_2, input_rel_3, input_rel_4], input_rel_1)
        input_rel_2 = self.fb_basic_block_5_2([input_rel_1, input_rel_3, input_rel_4], input_rel_2)
        input_rel_3 = self.fb_basic_block_5_3([input_rel_1, input_rel_2, input_rel_4], input_rel_3)
        input_rel_4 = self.fb_basic_block_5_4([input_rel_1, input_rel_2, input_rel_3], input_rel_4)
        input_rel_5 = self.fb_basic_block_5_5([input_rel_1, input_rel_2, input_rel_3, input_rel_4])

        return [input_rel_1, input_rel_2, input_rel_3, input_rel_4, input_rel_5]


class FullyAggregationBlock(nn.Module):
    def __init__(self):
        super(FullyAggregationBlock, self).__init__()

        # stage 1
        # 节点1+节点2上采样+节点上采样+节点4上采样
        self.fb_basic_block_1 = FusionBasicBlock(0, 3, 1, 32, in_channel_list=[32, 64, 128, 256, 512], cbam_block=True)
        # stage 2
        # 节点1下采样+节点2+节点3上采样+节点4上采样
        self.fb_basic_block_2 = FusionBasicBlock(1, 2, 2, 64, in_channel_list=[32, 64, 128, 256, 512], cbam_block=True)
        # stage 3
        # 节点1下采样+节点2下采样+节点3+节点4上采样
        self.fb_basic_block_3 = FusionBasicBlock(2, 1, 3, 128, in_channel_list=[32, 64, 128, 256, 512], cbam_block=True)
        # stage 4
        # 节点1下采样+节点2下采样+节点3下采样+节点4
        self.fb_basic_block_4 = FusionBasicBlock(3, 0, 4, 256, in_channel_list=[32, 64, 128, 256, 512], cbam_block=True)

    def forward(self, x):
        [l1, l2, l3, l4] = x
        l1 = self.fb_basic_block_1([l2, l3, l4], l1)
        l2 = self.fb_basic_block_2([l1, l3, l4], l2)
        l3 = self.fb_basic_block_3([l1, l2, l4], l3)
        l4 = self.fb_basic_block_4([l1, l2, l3], l4)

        return [l1, l2, l3, l4]


class UNetMRFG(nn.Module):
    def __init__(self):
        super().__init__()

        self.mr_block = MultiResolutionBlock()

        self.conv_block_1 = ResBlock(1, 32)
        self.down_sampling_1 = nn.MaxPool2d(2, 2)
        self.conv_block_2 = ResBlock(32, 64)
        self.down_sampling_2 = nn.MaxPool2d(2, 2)
        self.conv_block_3 = ResBlock(64, 128)
        self.down_sampling_3 = nn.MaxPool2d(2, 2)
        self.conv_block_4 = ResBlock(128, 256)
        self.down_sampling_4 = nn.MaxPool2d(2, 2)
        self.conv_block_5 = ResBlock(256, 512)

        self.fg_block = FullyAggregationBlock()

        self.up_sampling_1 = UpSamplingBlock(512, 256)
        self.conv_block_6 = ResBlock(512, 256)
        self.up_sampling_2 = UpSamplingBlock(256, 128)
        self.conv_block_7 = ResBlock(256, 128)
        self.up_sampling_3 = UpSamplingBlock(128, 64)
        self.conv_block_8 = ResBlock(128, 64)
        self.up_sampling_4 = UpSamplingBlock(64, 32)
        self.conv_block_9 = ResBlock(64, 32)

        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        [input_l1, input_l2, input_l3, input_l4, input_l5] = self.mr_block(x)

        conv_block_out_1 = self.conv_block_1(input_l1)
        conv_block_out_2 = self.conv_block_2(input_l2)
        conv_block_out_3 = self.conv_block_3(input_l3)
        conv_block_out_4 = self.conv_block_4(input_l4)
        conv_block_out_5 = self.conv_block_5(input_l5)

        [stage_1, stage_2, stage_3, stage_4] = self.fg_block(
            [conv_block_out_1, conv_block_out_2, conv_block_out_3, conv_block_out_4])

        conv_block_out_6 = self.conv_block_6(self.up_sampling_1(conv_block_out_5, stage_4))
        conv_block_out_7 = self.conv_block_7(self.up_sampling_2(conv_block_out_6, stage_3))
        conv_block_out_8 = self.conv_block_8(self.up_sampling_3(conv_block_out_7, stage_2))
        conv_block_out_9 = self.conv_block_9(self.up_sampling_4(conv_block_out_8, stage_1))

        out = self.out(conv_block_out_9)

        return out
