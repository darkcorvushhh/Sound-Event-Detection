import math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pooling_size: int = 2) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=pooling_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ConvBlockPooling1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pooling_size: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(pooling_size, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ConvBlockPooling2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pooling_size: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(1, pooling_size))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Crnn_Baseline(nn.Module):
    def __init__(self, num_freq: int, class_num: int, **kwargs) -> None:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           num_freq: int, mel frequency bins
        #           class_num: int, the number of output classes
        ##############################
        super(Crnn_Baseline, self).__init__()
        self.bn = nn.BatchNorm1d(num_freq)
        self.conv_block1 = ConvBlock(1, 16)
        self.conv_block2 = ConvBlock(16, 32)
        self.conv_block3 = ConvBlock(32, 64)
        self.birnn = nn.GRU(int(num_freq/8*64), 64, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, class_num)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x: Tensor) -> Tensor:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           x: [batch_size, time_steps, num_freq]
        # Return:
        #           frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        b, t, _ = x.shape  # (b=32, t=501, f=64)
        x = x.transpose(1, 2)  # (b, f, t)
        x = self.bn(x)  # (b, f, t)
        x = x.unsqueeze(1)  # (b, c=1, f=64, t)
        x = self.conv_block1(x)  # (b, c=16, f=32, t/2)
        x = self.conv_block2(x)  # (b, c=32, f=16, t/4)
        x = self.conv_block3(x)  # (b, c=64, f=8, int(t/8)=62)
        x = x.reshape((b, -1, int(t/8)))  # (b, c*f, t/8)
        x = x.transpose(1, 2)  # (b, t/8, 512)
        x, _ = self.birnn(x)  # (b, t/8, 128)
        x = self.fc(x)  # (b, t, class_num)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class Crnn_pooling1(nn.Module):
    def __init__(self, num_freq: int, class_num: int, **kwargs) -> None:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           num_freq: int, mel frequency bins
        #           class_num: int, the number of output classes
        ##############################
        super().__init__()
        # self.bn = nn.BatchNorm1d(num_freq)
        self.bn = nn.BatchNorm2d(1)
        self.conv_block1 = ConvBlockPooling1(1, 16, kernel_size=kwargs['ks'])
        self.conv_block2 = ConvBlockPooling1(16, 32, kernel_size=kwargs['ks'])
        self.conv_block3 = ConvBlockPooling1(32, 64, kernel_size=kwargs['ks'])
        self.birnn = nn.GRU(num_freq*8, 64, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, class_num)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x: Tensor) -> Tensor:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           x: [batch_size, time_steps, num_freq]
        # Return:
        #           frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        b, t, _ = x.shape  # (b=32, t=501, f=64)
        x = x.transpose(1, 2)  # (b, f, t)
        x = x.unsqueeze(1)  # (b, c=1, f=256, t)
        x = self.bn(x)  # (b, f, t)
        x = self.conv_block1(x)  # (b, c=16, f=128, t)
        x = self.conv_block2(x)  # (b, c=32, f=64, t)
        x = self.conv_block3(x)  # (b, c=64, f=32, t)
        x = x.reshape((b, -1, t))  # (b, c*f, t)
        x = x.transpose(1, 2)  # (b, t, 512)
        x, _ = self.birnn(x)  # (b, t, 1024)
        x = self.fc(x)  # (b, t, class_num)
        x = self.sigmoid(x).clamp(min=1e-7, max=1.)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class Crnn_Interpolate(nn.Module):
    def __init__(self, num_freq: int, class_num: int, **kwargs) -> None:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           num_freq: int, mel frequency bins
        #           class_num: int, the number of output classes
        ##############################
        super(Crnn_Interpolate, self).__init__()
        self.bn = nn.BatchNorm1d(num_freq)
        self.conv_block1 = ConvBlock(1, 16)
        self.conv_block2 = ConvBlock(16, 32)
        self.conv_block3 = ConvBlock(32, 64)
        self.birnn = nn.GRU(int(num_freq/8*64), 64, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, class_num)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x: Tensor) -> Tensor:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           x: [batch_size, time_steps, num_freq]
        # Return:
        #           frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        b, t, _ = x.shape  # (b=32, t=501, f=64)
        x = x.transpose(1, 2)  # (b, f, t)
        x = self.bn(x)  # (b, f, t)
        x = x.unsqueeze(1)  # (b, c=1, f=64, t)
        x = self.conv_block1(x)  # (b, c=16, f=32, t/2)
        x = self.conv_block2(x)  # (b, c=32, f=16, t/4)
        x = self.conv_block3(x)  # (b, c=64, f=8, int(t/8)=62)
        x = x.reshape((b, -1, int(t/8)))  # (b, c*f, t/8)
        x = x.transpose(1, 2)  # (b, t/8, 512)
        x, _ = self.birnn(x)  # (b, t/8, 128)
        x = self.fc(x)  # (b, t, class_num)
        x = self.sigmoid(x).clamp(min=1e-7, max=1.)
        x = nn.functional.interpolate(x.transpose(1, 2), size=t,
                                      mode='linear',
                                      align_corners=False).transpose(1, 2)  # (b, t, class_num)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class Crnn_LSTM(nn.Module):
    def __init__(self, num_freq: int, class_num: int, **kwargs) -> None:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           num_freq: int, mel frequency bins
        #           class_num: int, the number of output classes
        ##############################
        super(Crnn_LSTM, self).__init__()
        self.bn = nn.BatchNorm1d(num_freq)
        self.conv_block1 = ConvBlock(1, 16)
        self.conv_block2 = ConvBlock(16, 32)
        self.conv_block3 = ConvBlock(32, 64)
        self.birnn = nn.LSTM(int(num_freq/8*64), 64, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, class_num)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x: Tensor) -> Tensor:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           x: [batch_size, time_steps, num_freq]
        # Return:
        #           frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        b, t, _ = x.shape  # (b=32, t=501, f=64)
        x = x.transpose(1, 2)  # (b, f, t)
        x = self.bn(x)  # (b, f, t)
        x = x.unsqueeze(1)  # (b, c=1, f=64, t)
        x = self.conv_block1(x)  # (b, c=16, f=32, t/2)
        x = self.conv_block2(x)  # (b, c=32, f=16, t/4)
        x = self.conv_block3(x)  # (b, c=64, f=8, int(t/8)=62)
        x = x.reshape((b, -1, int(t/8)))  # (b, c*f, t/8)
        x = x.transpose(1, 2)  # (b, t/8, 512)
        x, _ = self.birnn(x)  # (b, t/8, 128)
        x = self.fc(x)  # (b, t, class_num)
        x = self.sigmoid(x).clamp(min=1e-7, max=1.)
        x = nn.functional.interpolate(x.transpose(1, 2), size=t,
                                      mode='linear',
                                      align_corners=False).transpose(1, 2)  # (b, t, class_num)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class Crnn_kernel9(nn.Module):
    def __init__(self, num_freq: int, class_num: int, **kwargs) -> None:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           num_freq: int, mel frequency bins
        #           class_num: int, the number of output classes
        ##############################
        super(Crnn_kernel9, self).__init__()
        self.bn = nn.BatchNorm1d(num_freq)
        self.conv_block1 = ConvBlock(1, 16, kernel_size=kwargs['ks'])
        self.conv_block2 = ConvBlock(16, 32, kernel_size=kwargs['ks'])
        self.conv_block3 = ConvBlock(32, 64, kernel_size=kwargs['ks'])
        self.birnn = nn.GRU(int(num_freq/8*64), 64, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, class_num)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x: Tensor) -> Tensor:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           x: [batch_size, time_steps, num_freq]
        # Return:
        #           frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        b, t, _ = x.shape  # (b=32, t=501, f=64)
        x = x.transpose(1, 2)  # (b, f, t)
        x = self.bn(x)  # (b, f, t)
        x = x.unsqueeze(1)  # (b, c=1, f=64, t)
        x = self.conv_block1(x)  # (b, c=16, f=32, t/2)
        x = self.conv_block2(x)  # (b, c=32, f=16, t/4)
        x = self.conv_block3(x)  # (b, c=64, f=8, int(t/8)=62)
        x = x.reshape((b, -1, int(t/8)))  # (b, c*f, t/8)
        x = x.transpose(1, 2)  # (b, t/8, 512)
        x, _ = self.birnn(x)  # (b, t/8, 128)
        x = self.fc(x)  # (b, t, class_num)
        x = self.sigmoid(x).clamp(min=1e-7, max=1.)
        x = nn.functional.interpolate(x.transpose(1, 2), size=t,
                                      mode='linear',
                                      align_corners=False).transpose(1, 2)  # (b, t, class_num)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class Crnn_256_interpolate(nn.Module):
    def __init__(self, num_freq: int, class_num: int, **kwargs) -> None:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           num_freq: int, mel frequency bins
        #           class_num: int, the number of output classes
        ##############################
        super().__init__()
        # self.bn = nn.BatchNorm1d(num_freq)
        self.bn = nn.BatchNorm2d(1)
        self.conv_block1 = ConvBlock(1, 16, kernel_size=kwargs['ks'])
        self.conv_block2 = ConvBlock(16, 32, kernel_size=kwargs['ks'])
        self.conv_block3 = ConvBlock(32, 64, kernel_size=kwargs['ks'])
        self.conv_block4 = ConvBlock(64, 128, kernel_size=kwargs['ks'])
        self.conv_block5 = ConvBlock(128, 256, kernel_size=kwargs['ks'])
        self.birnn = nn.GRU(num_freq*8, 512, num_layers=kwargs['nl'], bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024, class_num)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x: Tensor) -> Tensor:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           x: [batch_size, time_steps, num_freq]
        # Return:
        #           frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        b, t, _ = x.shape  # (b=32, t=501, f=64)
        x = x.transpose(1, 2)  # (b, f, t)
        x = x.unsqueeze(1)  # (b, c=1, f=256, t)
        x = self.bn(x)  # (b, f, t)
        x = self.conv_block1(x)  # (b, c=16, f=128, t/2)
        x = self.conv_block2(x)  # (b, c=32, f=64, t/4)
        x = self.conv_block3(x)  # (b, c=64, f=32, int(t/8)=62)
        x = self.conv_block4(x)  # (b, c=128, f=16, int(t/16))
        x = self.conv_block5(x)  # (b, c=256, f=8, int(t/32))
        x = x.reshape((b, -1, int(t/32)))  # (b, c*f, t/32)
        x = x.transpose(1, 2)  # (b, t/32, 512)
        x, _ = self.birnn(x)  # (b, t/32, 1024)
        x = self.fc(x)  # (b, t/32, class_num)
        x = self.sigmoid(x).clamp(min=1e-7, max=1.)
        x = nn.functional.interpolate(x.transpose(1, 2), size=t,
                                      mode='nearest').transpose(1, 2)  # (b, t, class_num)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class Crnn_256_pooling1(nn.Module):
    def __init__(self, num_freq: int, class_num: int, **kwargs) -> None:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           num_freq: int, mel frequency bins
        #           class_num: int, the number of output classes
        ##############################
        super().__init__()
        # self.bn = nn.BatchNorm1d(num_freq)
        self.bn = nn.BatchNorm2d(1)
        self.conv_block1 = ConvBlockPooling1(1, 16, kernel_size=kwargs['ks'])
        self.conv_block2 = ConvBlockPooling1(16, 32, kernel_size=kwargs['ks'])
        self.conv_block3 = ConvBlockPooling1(32, 64, kernel_size=kwargs['ks'])
        self.conv_block4 = ConvBlockPooling1(64, 128, kernel_size=kwargs['ks'])
        self.conv_block5 = ConvBlockPooling1(128, 256, kernel_size=kwargs['ks'])
        self.birnn = nn.GRU(num_freq*8, 512, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024, class_num)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x: Tensor) -> Tensor:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           x: [batch_size, time_steps, num_freq]
        # Return:
        #           frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        b, t, _ = x.shape  # (b=32, t=501, f=64)
        x = x.transpose(1, 2)  # (b, f, t)
        x = x.unsqueeze(1)  # (b, c=1, f=256, t)
        x = self.bn(x)  # (b, f, t)
        x = self.conv_block1(x)  # (b, c=16, f=128, t)
        x = self.conv_block2(x)  # (b, c=32, f=64, t)
        x = self.conv_block3(x)  # (b, c=64, f=32, t)
        x = self.conv_block4(x)  # (b, c=128, f=16, t)
        x = self.conv_block5(x)  # (b, c=256, f=8, t)
        x = x.reshape((b, -1, t))  # (b, c*f, t)
        x = x.transpose(1, 2)  # (b, t, 512)
        x, _ = self.birnn(x)  # (b, t, 1024)
        x = self.fc(x)  # (b, t, class_num)
        x = self.sigmoid(x).clamp(min=1e-7, max=1.)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class Crnn_256_pooling2(nn.Module):
    def __init__(self, num_freq: int, class_num: int, **kwargs) -> None:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           num_freq: int, mel frequency bins
        #           class_num: int, the number of output classes
        ##############################
        super().__init__()
        # self.bn = nn.BatchNorm1d(num_freq)
        self.bn = nn.BatchNorm2d(1)
        self.conv_block1 = ConvBlockPooling2(1, 16, kernel_size=kwargs['ks'])
        self.conv_block2 = ConvBlockPooling2(16, 32, kernel_size=kwargs['ks'])
        self.conv_block3 = ConvBlockPooling2(32, 64, kernel_size=kwargs['ks'])
        self.conv_block4 = ConvBlockPooling2(64, 128, kernel_size=kwargs['ks'])
        self.conv_block5 = ConvBlockPooling2(128, 256, kernel_size=kwargs['ks'])
        self.birnn = nn.GRU(num_freq*8, 512, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024, class_num)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x: Tensor) -> Tensor:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           x: [batch_size, time_steps, num_freq]
        # Return:
        #           frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        b, t, _ = x.shape  # (b=32, t=501, f=64)
        x = x.unsqueeze(1)  # (b, c=1, f=256, t)
        x = self.bn(x)  # (b, f, t)
        x = self.conv_block1(x)  # (b, c=16, f=128, t)
        x = self.conv_block2(x)  # (b, c=32, f=64, t)
        x = self.conv_block3(x)  # (b, c=64, f=32, t)
        x = self.conv_block4(x)  # (b, c=128, f=16, t)
        x = self.conv_block5(x)  # (b, c, t, f)
        x, _ = self.birnn(x.permute((0, 2, 1, 3)).flatten(2))  # (b, t, 1024)
        x = self.fc(x)  # (b, t, class_num)
        x = self.sigmoid(x).clamp(min=1e-7, max=1.)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class Crnn_256_interpolate_lstm(nn.Module):
    def __init__(self, num_freq: int, class_num: int, **kwargs) -> None:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           num_freq: int, mel frequency bins
        #           class_num: int, the number of output classes
        ##############################
        super().__init__()
        # self.bn = nn.BatchNorm1d(num_freq)
        self.bn = nn.BatchNorm2d(1)
        self.conv_block1 = ConvBlock(1, 16, kernel_size=kwargs['ks'])
        self.conv_block2 = ConvBlock(16, 32, kernel_size=kwargs['ks'])
        self.conv_block3 = ConvBlock(32, 64, kernel_size=kwargs['ks'])
        self.conv_block4 = ConvBlock(64, 128, kernel_size=kwargs['ks'])
        self.conv_block5 = ConvBlock(128, 256, kernel_size=kwargs['ks'])
        self.birnn = nn.LSTM(num_freq*8, 512, num_layers=kwargs['nl'], bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024, class_num)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x: Tensor) -> Tensor:
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #           x: [batch_size, time_steps, num_freq]
        # Return:
        #           frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        b, t, _ = x.shape  # (b=32, t=501, f=64)
        x = x.transpose(1, 2)  # (b, f, t)
        x = x.unsqueeze(1)  # (b, c=1, f=256, t)
        x = self.bn(x)  # (b, f, t)
        x = self.conv_block1(x)  # (b, c=16, f=128, t/2)
        x = self.conv_block2(x)  # (b, c=32, f=64, t/4)
        x = self.conv_block3(x)  # (b, c=64, f=32, int(t/8)=62)
        x = self.conv_block4(x)  # (b, c=128, f=16, int(t/16))
        x = self.conv_block5(x)  # (b, c=256, f=8, int(t/32))
        x = x.reshape((b, -1, int(t/32)))  # (b, c*f, t/32)
        x = x.transpose(1, 2)  # (b, t/32, 512)
        x, _ = self.birnn(x)  # (b, t/32, 1024)
        x = self.fc(x)  # (b, t/32, class_num)
        x = self.sigmoid(x).clamp(min=1e-7, max=1.)
        x = nn.functional.interpolate(x.transpose(1, 2), size=t,
                                      mode='nearest').transpose(1, 2)  # (b, t, class_num)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }