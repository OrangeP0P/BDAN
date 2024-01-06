import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from SVG import SVG
import math
# import ot

class depthwise_separable_conv(nn.Module):  # 深度可分离卷积
    def __init__(self, nin, nout, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=(1, kernel_size), padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nout, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class TEGDAN_1(nn.Module):
    def __init__(self, act_func):
        super(TEGDAN_1, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112
        self.count = 0
        self.ET = torch.from_numpy(np.load('ET.npy', allow_pickle=True)).cuda()  # 读取权重

        self.Sequence1 = nn.Sequential()
        self.Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.Sequence1.add_module('A-ELU1', nn.ReLU())
        self.Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.Sequence2 = nn.Sequential()
        self.Sequence2.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1 * 2, (3, 1), stride=(1, 1)))
        self.Sequence2.add_module('B-ELU1', nn.ReLU())
        self.Sequence2.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.Sequence3 = nn.Sequential()
        self.Sequence3.add_module('C-Conv1', nn.Conv2d(self.F1*2, self.F1*2, (1, 10), stride=(1, 1)))
        self.Sequence3.add_module('C-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence3.add_module('C-ELU1', nn.ReLU())
        self.Sequence3.add_module('C-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence3.add_module('C-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.Sequence4 = nn.Sequential()
        self.Sequence4.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (self.channel, 1), stride=(1, 1)))
        self.Sequence4.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence4.add_module('S-ELU1', nn.ReLU())
        self.Sequence4.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 4))

    def forward(self, source, target, bridging_domain):
        intra_source_ele_loss = 0
        intra_target_ele_loss = 0
        intra_ele_loss = 0

        source_all = self.Sequence1(source)
        source_all = F.pad(source_all, (1, 1, 2, 0))
        source_all = self.Sequence2(source_all)
        source_all = self.Sequence3(source_all)

        if self.training:
            target_all = self.Sequence1(target)
            target_all = F.pad(target_all, (1, 1, 2, 0))
            target_all = self.Sequence2(target_all)
            target_all = self.Sequence3(target_all)

            _bridging_domain = torch.mean((source_all + target_all), dim=0)
            _bridging_domain = _bridging_domain.unsqueeze(0)
            bridging_domain = (bridging_domain + _bridging_domain)/2
            std_bridge = torch.std(bridging_domain).item()
            noise_mean = 0
            noise_std = 0.001 * std_bridge
            noise = torch.normal(noise_mean, noise_std, size=(28, 16, 118, 11)).cuda()
            bridging_domain_re = torch.repeat_interleave(bridging_domain.data, repeats=28, dim=0)
            bridging_domain_re_noise = bridging_domain_re + noise

            source_ele_1 = self.SFC1(source_all)  # 3D全连接层1
            source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2
            source_all = self.Sequence4(source_ele_2)

            # 源域用户内电极损失 第一层
            _s0, _s1, _s2 = source_ele_1.shape[:3]
            SE1 = source_ele_1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            BE = bridging_domain_re.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)
            _s = SE1 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _s = _s @ _s.transpose(-1,
                                   -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _ms1 = _s.mean(dim=-1)
            _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
            _ms3 = _ms2  # 赋予权重
            _ms4 = _ms3.sum()
            intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)

            # 源域用户内电极损失 第二层
            _s0, _, _s2 = source_ele_2.shape[:3]
            SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
            _s = SE2 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _s = _s @ _s.transpose(-1,
                                   -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _ms1 = _s.mean(dim=-1)
            _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
            _ms3 = _ms2  # 赋予权重
            _ms4 = _ms3.sum()
            intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)

            target_ele_1 = self.SFC1(target_all)  # 3D全连接层1
            target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2
            target_all = self.Sequence4(target_ele_2)

            # 目标域 Stage 1 第一层
            _t0, _t1, _t2 = target_ele_1.shape[:3]
            TE1 = target_ele_1.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            _t = TE1 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)  # [s2, s2, s0, s0]
            _mt3 = _mt2  # 赋予权重
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            # 目标域 Stage 2 第二层
            _t0, _t1, _t2 = target_ele_2.shape[:3]
            TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            _t = TE2 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)  # [s2, s2, s0, s0]
            _mt3 = _mt2  # 赋予权重
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            intra_ele_loss += intra_source_ele_loss + intra_target_ele_loss # 总 loss

            # 拉伸成条形处理
            s0, s1, s2, s3 = target_all.shape[:4]  # 读取张量大小
            source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]

            source_all = self.FC1(source_all)

            output = self.F_FC1(source_all)

        return output, intra_ele_loss, intra_source_ele_loss, intra_target_ele_loss, bridging_domain, std_bridge

class TEGDAN_2(nn.Module):
    def __init__(self, act_func):
        super(TEGDAN_2, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112
        self.count = 0
        self.ET = torch.from_numpy(np.load('ET.npy', allow_pickle=True)).cuda()  # 读取权重

        self.Sequence1 = nn.Sequential()
        self.Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, track_running_stats=True))
        self.Sequence1.add_module('A-ELU1', nn.ReLU())
        self.Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.Sequence2 = nn.Sequential()
        self.Sequence2.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1 * 2, (3, 1), stride=(1, 1)))
        self.Sequence2.add_module('B-ELU1', nn.ReLU())
        self.Sequence2.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.Sequence3 = nn.Sequential()
        self.Sequence3.add_module('C-Conv1', nn.Conv2d(self.F1*2, self.F1*2, (1, 10), stride=(1, 1)))
        self.Sequence3.add_module('C-Norm1', nn.BatchNorm2d(self.F1*2, track_running_stats=True))
        self.Sequence3.add_module('C-ELU1', nn.ReLU())
        self.Sequence3.add_module('C-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence3.add_module('C-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.Sequence4 = nn.Sequential()
        self.Sequence4.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (self.channel, 1), stride=(1, 1)))
        self.Sequence4.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, track_running_stats=True))
        self.Sequence4.add_module('S-ELU1', nn.ReLU())
        self.Sequence4.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 4))

    def forward(self, source, target, bridging_domain):
        intra_source_ele_loss = 0
        intra_target_ele_loss = 0
        intra_ele_loss = 0
        std_bridge = 0

        source_all = self.Sequence1(source)
        source_all = F.pad(source_all, (1, 1, 2, 0))
        source_all = self.Sequence2(source_all)
        source_all = self.Sequence3(source_all)

        source_ele_1 = self.SFC1(source_all)  # 3D全连接层1
        source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2
        source_all = self.Sequence4(source_ele_2)

        _bridging_domain = torch.mean((source_all + source_all), dim=0)
        _bridging_domain = _bridging_domain.unsqueeze(0)
        bridging_domain = (bridging_domain + _bridging_domain) / 2
        std_bridge = torch.std(bridging_domain).item()

        if self.training:
            target_all = self.Sequence1(target)
            target_all = F.pad(target_all, (1, 1, 2, 0))
            target_all = self.Sequence2(target_all)
            target_all = self.Sequence3(target_all)

            target_ele_1 = self.SFC1(target_all)  # 3D全连接层1
            target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2
            target_all = self.Sequence4(target_ele_2)

        # 拉伸成条形处理
        s0, s1, s2, s3 = source_all.shape[:4]  # 读取张量大小
        source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]
        source_all = self.FC1(source_all)

        output = self.F_FC1(source_all)

        return output, intra_ele_loss, intra_source_ele_loss, intra_target_ele_loss, bridging_domain, std_bridge

class EEGNet(nn.Module):  # Net1: EEGNet
    def __init__(self):
        super(EEGNet, self).__init__()
        self.Kernel = 80
        self.F1 = 8
        self.DF = 2
        self.Channel = 118
        self.Class = 4
        self.mapsize = 160

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.25))

        self.Extractor.add_module('c-3', depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 128))
        self.class_classifier.add_module('fb-1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('fc-2', nn.Linear(128, 64))
        self.class_classifier.add_module('fb-2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('fc-3', nn.Linear(64, self.Class))

    def forward(self, source_data, target_data, bridging_domain):
        loss_1 = torch.from_numpy(np.array(0)).cuda()
        loss_2 = torch.from_numpy(np.array(0)).cuda()
        loss_3 = torch.from_numpy(np.array(0)).cuda()
        feature = self.Extractor(source_data)
        feature = feature.view(-1, self.mapsize)
        _bridging_domain = torch.mean((feature + feature), dim=0)
        _bridging_domain = _bridging_domain.unsqueeze(0)
        bridging_domain = (bridging_domain + _bridging_domain) / 2
        std_bridge = torch.std(bridging_domain).item()

        class_output = self.class_classifier(feature)

        return class_output, loss_1, loss_2, loss_3, bridging_domain, std_bridge

class TEGDAN_4(nn.Module):
    def __init__(self, act_func):
        super(TEGDAN_4, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112
        self.count = 0
        self.ET = torch.from_numpy(np.load('ET.npy', allow_pickle=True)).cuda()  # 读取权重

        self.Sequence1 = nn.Sequential()
        self.Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.Sequence1.add_module('A-ELU1', nn.ReLU())
        self.Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.Sequence2 = nn.Sequential()
        self.Sequence2.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1 * 2, (3, 1), stride=(1, 1)))
        self.Sequence2.add_module('B-ELU1', nn.ReLU())
        self.Sequence2.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.Sequence3 = nn.Sequential()
        self.Sequence3.add_module('C-Conv1', nn.Conv2d(self.F1*2, self.F1*2, (1, 10), stride=(1, 1)))
        self.Sequence3.add_module('C-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence3.add_module('C-ELU1', nn.ReLU())
        self.Sequence3.add_module('C-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence3.add_module('C-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.Sequence4 = nn.Sequential()
        self.Sequence4.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (self.channel, 1), stride=(1, 1)))
        self.Sequence4.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence4.add_module('S-ELU1', nn.ReLU())
        self.Sequence4.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 4))

    def forward(self, source, target, bridging_domain):
        intra_source_ele_loss = 0
        intra_target_ele_loss = 0
        intra_ele_loss = 0

        source_all = self.Sequence1(source)
        source_all = F.pad(source_all, (1, 1, 2, 0))
        source_all = self.Sequence2(source_all)
        source_all = self.Sequence3(source_all)

        if self.training:
            target_all = self.Sequence1(target)
            target_all = F.pad(target_all, (1, 1, 2, 0))
            target_all = self.Sequence2(target_all)
            target_all = self.Sequence3(target_all)

            _bridging_domain = torch.mean((source_all + target_all), dim=0)
            _bridging_domain = _bridging_domain.unsqueeze(0)
            bridging_domain = (bridging_domain + _bridging_domain)/2

            # std_bridge = torch.std(source_all-target_all).item()  # 计算该batch全局数据的标准差std_bridge
            # noise_mean = 0  # 设置高斯噪声均值为0
            # noise_std = 0.1 * std_bridge  # 设置高斯噪声标准差为 0.001 * std_bridge 记录0.001
            std_bridge = torch.std(source_all-target_all).item()  # 计算该batch数据的标准差std_bridge
            noise_mean = 0  # 设置高斯噪声均值为0
            noise_std = 0.12 * std_bridge  # 设置高斯噪声标准差为 0.001 * std_bridge 记录0.001
            noise = torch.normal(noise_mean, noise_std, size=(28, 16, 118, 11)).cuda()  # 产生高斯噪声
            bridging_domain_re = torch.repeat_interleave(bridging_domain.data, repeats=28, dim=0)
            bridging_domain_re_noise = bridging_domain_re + noise  # 在生成域数据上加高斯噪声进行扰动

            source_ele_1 = self.SFC1(source_all)  # 3D全连接层1
            source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2
            source_all = self.Sequence4(source_ele_2)

            # 源域用户内电极损失 第一层
            _s0, _s1, _s2 = source_ele_1.shape[:3]
            SE1 = source_ele_1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            BE = bridging_domain_re_noise.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)
            _s = SE1 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _s = _s @ _s.transpose(-1,
                                   -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _ms1 = _s.mean(dim=-1)
            _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
            _ms3 = _ms2  # 赋予权重
            _ms4 = _ms3.sum()
            intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)

            # 源域用户内电极损失 第二层
            _s0, _, _s2 = source_ele_2.shape[:3]
            SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
            _s = SE2 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _s = _s @ _s.transpose(-1,
                                   -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _ms1 = _s.mean(dim=-1)
            _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
            _ms3 = _ms2  # 赋予权重
            _ms4 = _ms3.sum()
            intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)

            target_ele_1 = self.SFC1(target_all)  # 3D全连接层1
            target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2
            target_all = self.Sequence4(target_ele_2)

            # 目标域 Stage 1 第一层
            _t0, _t1, _t2 = target_ele_1.shape[:3]
            TE1 = target_ele_1.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            _t = TE1 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)  # [s2, s2, s0, s0]
            _mt3 = _mt2  # 赋予权重
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            # 目标域 Stage 2 第二层
            _t0, _t1, _t2 = target_ele_2.shape[:3]
            TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            _t = TE2 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)  # [s2, s2, s0, s0]
            _mt3 = _mt2  # 赋予权重
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            intra_ele_loss += intra_source_ele_loss + intra_target_ele_loss # 总 loss

            # 拉伸成条形处理
            s0, s1, s2, s3 = target_all.shape[:4]  # 读取张量大小
            source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]

            source_all = self.FC1(source_all)
            output = self.F_FC1(source_all)

        return output, intra_ele_loss, intra_source_ele_loss, intra_target_ele_loss, bridging_domain, std_bridge

class TEGDAN_5(nn.Module):
    def __init__(self, act_func):
        super(TEGDAN_5, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112
        self.count = 0
        self.ET = torch.from_numpy(np.load('ET.npy', allow_pickle=True)).cuda()  # 读取权重

        self.Sequence1 = nn.Sequential()
        self.Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.Sequence1.add_module('A-ELU1', nn.ReLU())
        self.Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.Sequence2 = nn.Sequential()
        self.Sequence2.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1 * 2, (3, 1), stride=(1, 1)))
        self.Sequence2.add_module('B-ELU1', nn.ReLU())
        self.Sequence2.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.Sequence3 = nn.Sequential()
        self.Sequence3.add_module('C-Conv1', nn.Conv2d(self.F1*2, self.F1*2, (1, 10), stride=(1, 1)))
        self.Sequence3.add_module('C-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence3.add_module('C-ELU1', nn.ReLU())
        self.Sequence3.add_module('C-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence3.add_module('C-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.Sequence4 = nn.Sequential()
        self.Sequence4.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (self.channel, 1), stride=(1, 1)))
        self.Sequence4.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence4.add_module('S-ELU1', nn.ReLU())
        self.Sequence4.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 4))
        self.svg = SVG(cuda=True,
                     sigma_rotate_theta=0.314/10000,  # Standard deviation of the theta rotation angle
                     sigma_rotate_phi=0.1/10000,  # Standard deviation of the phi rotation angle
                     sigma_stretch_theta=0.1/10000,  # Standard deviation of the theta stretching angle
                     sigma_stretch_phi=0.05/10000,  # Standard deviation of the phi stretching angle
                     sigma_pos=0.05/10000,  # Standard deviation for random electrode placement
                     att_k=-4,
                     cap='BCIIIIIV2A', probability_flip=0)

    def forward(self, source, target, bridging_domain):
        intra_source_ele_loss = 0
        intra_target_ele_loss = 0
        intra_ele_loss = 0

        source_all = self.Sequence1(source)
        source_all = F.pad(source_all, (1, 1, 2, 0))
        source_all = self.Sequence2(source_all)
        source_all = self.Sequence3(source_all)

        if self.training:
            target_all = self.Sequence1(target)
            target_all = F.pad(target_all, (1, 1, 2, 0))
            target_all = self.Sequence2(target_all)
            target_all = self.Sequence3(target_all)

            _bridging_domain = torch.mean((source_all + target_all), dim=0)
            _bridging_domain = _bridging_domain.unsqueeze(0)
            bridging_domain = (bridging_domain + _bridging_domain)/2

            std_bridge = 0  # 计算该batch数据的标准差std_bridge

            bridging_domain_re = torch.empty(28, 16, 118, 11).cuda()
            # 使用循环填充预分配的张量
            for i in range(28):
                bridging_domain_re[i], isFlip = self.svg.transform(bridging_domain.data, mode='Auto')

            source_ele_1 = self.SFC1(source_all)  # 3D全连接层1
            source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2
            source_all = self.Sequence4(source_ele_2)

            # 源域用户内电极损失 第一层
            _s0, _s1, _s2 = source_ele_1.shape[:3]
            SE1 = source_ele_1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            BE = bridging_domain_re.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)
            _s = SE1 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _s = _s @ _s.transpose(-1,
                                   -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _ms1 = _s.mean(dim=-1)
            _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
            _ms3 = _ms2  # 赋予权重
            _ms4 = _ms3.sum()
            intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)

            # 源域用户内电极损失 第二层
            _s0, _, _s2 = source_ele_2.shape[:3]
            SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
            _s = SE2 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _s = _s @ _s.transpose(-1,
                                   -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _ms1 = _s.mean(dim=-1)
            _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
            _ms3 = _ms2  # 赋予权重
            _ms4 = _ms3.sum()
            intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)

            target_ele_1 = self.SFC1(target_all)  # 3D全连接层1
            target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2
            target_all = self.Sequence4(target_ele_2)

            # 目标域 Stage 1 第一层
            _t0, _t1, _t2 = target_ele_1.shape[:3]
            TE1 = target_ele_1.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            _t = TE1 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)  # [s2, s2, s0, s0]
            _mt3 = _mt2  # 赋予权重
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            # 目标域 Stage 2 第二层
            _t0, _t1, _t2 = target_ele_2.shape[:3]
            TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            _t = TE2 - BE.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)  # [s2, s2, s0, s0]
            _mt3 = _mt2  # 赋予权重
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            intra_ele_loss += intra_source_ele_loss + intra_target_ele_loss # 总 loss

            # 拉伸成条形处理
            s0, s1, s2, s3 = target_all.shape[:4]  # 读取张量大小
            source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]

            source_all = self.FC1(source_all)
            output = self.F_FC1(source_all)

        return output, intra_ele_loss, intra_source_ele_loss, intra_target_ele_loss, bridging_domain, std_bridge

'''最大均值差异计算函数'''
def mmd_rbf(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

'''最大均值差异计算函数'''
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

# def wasserstein_distance(source_data, target_data, reg=1e-1):
#     batch_size, num_features = source_data.shape
#     source_data_expanded = source_data.unsqueeze(1).repeat(1, batch_size, 1)
#     target_data_expanded = target_data.unsqueeze(0).repeat(batch_size, 1, 1)
#     pairwise_distances = torch.cdist(source_data_expanded, target_data_expanded, p=2)
#
#     # No need to index ot.sinkhorn2's result
#     wasserstein_distances = torch.tensor(
#         [ot.sinkhorn2([], [], pairwise_distances[i].detach().cpu().numpy(), reg=reg) for i in range(batch_size)]
#     )
#     return wasserstein_distances

'''选取激活函数类型'''
def choose_act_func(act_name):
    if act_name == 'elu':
        return nn.ELU()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'lrelu':
        return nn.LeakyReLU()
    else:
        raise TypeError('activation_function type not defined.')

'''处理预定义网络和训练参数'''
def handle_param(args, net):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'rmsp':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    else:
        raise TypeError('optimizer type not defined.')
    if args.loss_function == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    else:
        raise TypeError('loss_function type not defined.')
    return optimizer, loss_function

'''选取网络和激活函数'''
def choose_net(args):
    if args.model == 'TEGDAN_1':
        return {
        'elu': [TEGDAN_1('relu')]
        # 'relu': [EEGNet('relu')],
        # 'lrelu': [EEGNet('lrelu')],
        }
    elif args.model == 'TEGDAN_2':
        return {
        'elu': [TEGDAN_2('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'TEGDAN_3':
        return {
        'elu': [TEGDAN_3('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'TEGDAN_4':
        return {
        'elu': [TEGDAN_4('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'EEGNet':
        return {
        'elu': [EEGNet()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'TEGDAN_5':
        return {
        'elu': [TEGDAN_5('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net9':
        return {
        'elu': [EDAN('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net10':
        return {
        'elu': [PSDAN('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net1F':
        return {
        'elu': [EEGNet_FLOPs()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net2F':
        return {
        'elu': [DeepConvNet_FLOPs()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net3F':
        return {
        'elu': [DDC_FLOPs()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net4F':
        return {
        'elu': [DeepCoral_FLOPs()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net7F':
        return {
        'elu': [IA_EDAN_FLOPs('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net8F':
        return {
        'elu': [IE_EDAN_FLOPs('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net9F':
        return {
        'elu': [EDAN_Flops('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    else:
        raise TypeError('model type not defined.')