import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell, Sign



class EncoderCell(nn.Module):
    def __init__(self,input_channels):
        super(EncoderCell, self).__init__()

        self.conv = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = ConvLSTMCell(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, input, hidden1, hidden2, hidden3):
        x = self.conv(input)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]



        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self, bottleneck):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, bottleneck, kernel_size=1, bias=False)
        #elf.conv = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        self.tanh =nn.Tanh()
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        #feat = self.conv2(feat)
        x = torch.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    def __init__(self,bottleneck, output_channels):
        super(DecoderCell, self).__init__()
        
        #self.conv1= nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1= nn.Conv2d(bottleneck, 512, kernel_size=1, stride=1, padding=0, bias=False)


        self.rnn1 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.rnn4 = ConvLSTMCell(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.conv2 = nn.Conv2d(
            32, output_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input, hidden1, hidden2, hidden3, hidden4):

        x = self.conv1(input)
        #x = self.conv2_dec(input)
      
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)

        hidden4 = self.rnn4(x, hidden4)
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)

        x = torch.tanh(self.conv2(x))/2
        return x, hidden1, hidden2, hidden3, hidden4


class GainFactor(nn.Module):
    def __init__(self):
        super(GainFactor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2,padding=1,  bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3,stride=2,padding=1,  bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2,padding=1,  bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2,padding=1,  bias=True)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=2, stride=2,padding=0,  bias=True)
                 

    def forward(self, input):
        g = self.conv1(input)
        g = F.elu(g)
        g = self.conv2(g)
        g = F.elu(g)
        g = self.conv3(g)
        g = F.elu(g)
        g = self.conv4(g)
        g = F.elu(g)
        g = self.conv5(g)
        g = F.elu(g)
        g=g+2
        return g
 
