import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fusion_strategy


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out


# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []

        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


# Inception Block
class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(InceptionBlock, self).__init__()
        # Số lượng kênh đầu ra cho từng nhánh trong khối inception
        branch_out_channels = int(out_channels / 4)  # Chia đều số kênh đầu ra cho 4 nhánh

        # Nhánh 1: Convolution với kernel size = 1
        self.branch1x1 = ConvLayer(in_channels, branch_out_channels, kernel_size=1, stride=stride)

        # Nhánh 2: Convolution với kernel size = 3
        self.branch3x3 = nn.Sequential(
            ConvLayer(in_channels, branch_out_channels, kernel_size=1, stride=1),
            ConvLayer(branch_out_channels, branch_out_channels, kernel_size=3, stride=stride)
        )

        # Nhánh 3: Convolution với kernel size = 5
        self.branch5x5 = nn.Sequential(
            ConvLayer(in_channels, branch_out_channels, kernel_size=1, stride=1),
            ConvLayer(branch_out_channels, branch_out_channels, kernel_size=5, stride=stride)
        )

        # Nhánh 4: MaxPooling + Convolution 1x1
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
            ConvLayer(in_channels, branch_out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # Tính toán đầu ra của từng nhánh
        out1 = self.branch1x1(x)
        out2 = self.branch3x3(x)
        out3 = self.branch5x5(x)
        out4 = self.branch_pool(x)

        # Ghép nối đầu ra của tất cả các nhánh
        out = torch.cat([out1, out2, out3, out4], dim=1)  # Nối dọc theo chiều kênh (dim=1)
        return out


# Multi Inception Block
class MultiInceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, n):
        super(MultiInceptionBlock, self).__init__()
        layers = []
        for i in range(n):
            # Tính toán số kênh đầu ra cho mỗi khối Inception
            current_out_channels = out_channels // (2 ** (n - i - 1)) if i < n - 1 else out_channels
            layers.append(InceptionBlock(in_channels if i == 0 else previous_out_channels, current_out_channels, kernel_size, stride))
            previous_out_channels = current_out_channels
        self.multi_inception = nn.Sequential(*layers)

    def forward(self, x):
        return self.multi_inception(x)


# DenseResidualBlock
class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseResidualBlock, self).__init__()

        # Nhánh 1: Conv 1x1
        self.branch1 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)

        # Nhánh 2: DenseNet nhánh với concat
        self.branch2_layers = nn.ModuleList([
            ConvLayer(in_channels + i * (out_channels // 4), out_channels // 4, kernel_size=3, stride=1)
            for i in range(4)
        ])
        self.branch2_transition = nn.Conv2d(in_channels + 4 * (out_channels // 4), out_channels, kernel_size=1, stride=1)

        # Nhánh 3: Conv 1x1 -> Conv 3x3 -> Conv 1x1
        self.branch3 = nn.Sequential(
            ConvLayer(in_channels, out_channels // 2, kernel_size=1, stride=1),
            ConvLayer(out_channels // 2, out_channels // 2, kernel_size=3, stride=1),
            ConvLayer(out_channels // 2, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # Tính toán đầu ra từ nhánh 1
        out1 = self.branch1(x)

        # Tính toán đầu ra từ nhánh 2 theo kiểu DenseNet
        branch2_outs = [x]
        for layer in self.branch2_layers:
            new_out = layer(torch.cat(branch2_outs, dim=1))
            branch2_outs.append(new_out)
        out2 = self.branch2_transition(torch.cat(branch2_outs, dim=1))

        # Tính toán đầu ra từ nhánh 3
        out3 = self.branch3(x)

        # Cộng các đầu ra từ 3 nhánh
        out = out1 + out2 + out3

        # Áp dụng hàm kích hoạt ReLU
        out = F.relu(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()

        growth_rate = out_channels // 4  # Số lượng kênh tăng trưởng ở mỗi layer

        # Layer đầu tiên: Conv 1x1
        self.conv1 = ConvLayer(in_channels, growth_rate, kernel_size=1, stride=1)

        # Các layer tiếp theo với kernel_size và DenseNet connection
        self.conv2 = ConvLayer(in_channels + growth_rate, growth_rate, kernel_size=kernel_size, stride=stride)
        self.conv3 = ConvLayer(in_channels + 2 * growth_rate, growth_rate, kernel_size=kernel_size, stride=stride)
        self.conv4 = ConvLayer(in_channels + 3 * growth_rate, growth_rate, kernel_size=kernel_size, stride=stride)

        # Layer cuối cùng: Conv 1x1 để tạo đầu ra cuối cùng
        self.conv_out = nn.Conv2d(in_channels + 4 * growth_rate, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        outputs = [x]  # Danh sách lưu các đầu ra của các layer trước

        # Layer 1
        out1 = self.conv1(x)
        outputs.append(out1)

        # Layer 2
        out2 = self.conv2(torch.cat(outputs, dim=1))
        outputs.append(out2)

        # Layer 3
        out3 = self.conv3(torch.cat(outputs, dim=1))
        outputs.append(out3)

        # Layer 4
        out4 = self.conv4(torch.cat(outputs, dim=1))
        outputs.append(out4)

        # Concat toàn bộ đầu ra và đi qua lớp Conv 1x1 cuối cùng
        out = torch.cat(outputs, dim=1)
        out = self.conv_out(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResBlock, self).__init__()
        mid_channels = out_channels // 2  # Số kênh đầu ra của conv1

        # Layer 1: Conv 1x1 (Shortcut)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Conv kernel_size x kernel_size với số kênh đầu ra là out_channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer 3: Conv kernel_size x kernel_size với số kênh giữ nguyên là out_channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        
        # Layer 4: Conv 1x1 cuối cùng để giữ nguyên số kênh out_channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Shortcut từ kết quả của Layer 1
        shortcut = self.conv1(x)
        
        # Residual path qua Layer 2 và Layer 3
        out = self.conv2(shortcut)
        out = self.conv3(out)
        
        # Cộng shortcut với kết quả Layer 3
        out += shortcut
        
        # Qua Layer 4 (Conv 1x1 cuối)
        out = self.conv4(out)
        
        # Kích hoạt ReLU
        out = F.relu(out)
        return out


# NestFuse network - light, no desnse
class NestFuse_autoencoder(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(NestFuse_autoencoder, self).__init__()
        self.deepsupervision = deepsupervision
        block_decoder = DenseBlock_light
        block_encoder = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block_encoder(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block_encoder(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block_encoder(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block_encoder(nb_filter[2], nb_filter[3], kernel_size, 1)
        # self.DB5_0 = block(nb_filter[3], nb_filter[4], kernel_size, 1)

        # decoder
        self.DB1_1 = block_decoder(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block_decoder(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block_decoder(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        self.DB1_2 = block_decoder(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block_decoder(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)

        self.DB1_3 = block_decoder(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def fusion(self, en1, en2, p_type):
        # attention weight
        fusion_function = fusion_strategy.attention_fusion_weight

        f1_0 = fusion_function(en1[0], en2[0], p_type)
        f2_0 = fusion_function(en1[1], en2[1], p_type)
        f3_0 = fusion_function(en1[2], en2[2], p_type)
        f4_0 = fusion_function(en1[3], en2[3], p_type)
        return [f1_0, f2_0, f3_0, f4_0]

    def decoder_train(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):

        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]