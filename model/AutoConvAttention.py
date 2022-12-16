import numpy as np
import numpy.random
import math
import torch
import torch.utils.data as Data
import torch.nn as nn
class CBAM(nn.Module):
    def __init__(self, in_channels, out_channels,Feature_Num,Out_Feature_size):
        super(CBAM, self).__init__()
        self.Features_ = nn.Sequential(
            nn.Linear(Feature_Num, Out_Feature_size),
            nn.BatchNorm1d(Out_Feature_size),
            nn.ReLU()
        )
        self.Max_Pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1)
        )
        self.Avg_Pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.Relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.Conv3 = nn.Conv2d(2, 1, 3, 1, 1)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, Features,Out_Feature_size):
        Feature = self.Features_(Features)
        Feature = Feature.view(Features.size(0), 3, int(math.sqrt(Out_Feature_size)),
                               int(math.sqrt(Out_Feature_size)))
        ##ChannelAttention
        Max_Pools = self.Max_Pool(Feature)
        Max_Pools = self.conv1(Max_Pools)
        Max_Pools = self.Relu(Max_Pools)
        Max_Pools = self.conv2(Max_Pools)
        Avg_Pools = self.Avg_Pool(Feature)
        Avg_Pools = self.conv1(Avg_Pools)
        Avg_Pools = self.Relu(Avg_Pools)
        Avg_Pools = self.conv2(Avg_Pools)
        Out_Feature = torch.multiply(Feature, self.sigmoid(Max_Pools + Avg_Pools))
        ##SpatialAttention
        Avg_Pool = torch.mean(Out_Feature, dim=1, keepdim=True)
        Max_Pool, _ = torch.max(Out_Feature, dim=1, keepdim=True)
        Cat_Features = torch.cat([Avg_Pool, Max_Pool], dim=1)
        Feature_New = self.Conv3(Cat_Features)
        Out = torch.multiply(Out_Feature, self.sigmoid2(Feature_New)) + Feature
        Out = Out.view(Features.size(0), -1)
        return Out
class AttenDAE(nn.Module):
    def __init__(self,Linears_F,Linears_L):
        super(AttenDAE, self).__init__()
        self.DAE_Model_Encoder=self.create_Enconder_layers(Linears_F)
        self.DAE_Model_Decoder=self.create_Deconder_layers(Linears_L)
        self.CBAM_Model= CBAM(in_channels=3, out_channels=16, Feature_Num=Linears_F[-1], Out_Feature_size=Linears_F[-1])

    def create_Enconder_layers(self, Linears_Ls):
        layers = []
        for x in range(0, len(Linears_Ls) - 1):
            layers += [
                nn.Linear(Linears_Ls[x], Linears_Ls[x + 1]),
                nn.BatchNorm1d(Linears_Ls[x + 1]),
                nn.ReLU()]
        return nn.Sequential(*layers)

    def create_Deconder_layers(self, Linears_Fs):
        layers = []
        for x in range(0, len(Linears_Fs) - 2):
            layers += [
                nn.Linear(Linears_Fs[x], Linears_Fs[x + 1]),
                nn.BatchNorm1d(Linears_Fs[x + 1]),
                nn.ReLU()]
        layers+=[
            nn.Linear(Linears_Fs[x+1], Linears_Fs[x + 2]),
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)

    def forward(self, Features):
        encoder=self.DAE_Model_Encoder(Features)

        Feature_num=math.ceil(encoder.shape[1]/3)
        #
        atten_encoder=self.CBAM_Model(encoder,Feature_num)
        decoder=self.DAE_Model_Decoder(atten_encoder)

        return encoder,decoder
class Model_(nn.Module):
    def __init__(self,Linears_F,Linears_L,Out_Feature_size):
        super(Model_, self).__init__()
        self.DeePL= AttenDAE(Linears_F=Linears_F, Linears_L=Linears_L)
        self.PreD=nn.Sequential(
            nn.Linear(Out_Feature_size,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,2),
            nn.Softmax(dim=1)

        )
    def forward(self, Feature):
        atten_encoder,decoder=self.DeePL(Feature)
        Pre_=self.PreD(atten_encoder)
        return atten_encoder,decoder,Pre_

