import math
import numpy as np
import numpy.random
import torch
import torch.nn as nn
torch.manual_seed(2021)  # To initialize the seed with the same randomness to form the same random effect
np.random.seed(2021)

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
        ##CBAM_ChannelAttention
        Max_Pools = self.Max_Pool(Feature)
        Max_Pools = self.conv1(Max_Pools)
        Max_Pools = self.Relu(Max_Pools)
        Max_Pools = self.conv2(Max_Pools)
        Avg_Pools = self.Avg_Pool(Feature)
        Avg_Pools = self.conv1(Avg_Pools)
        Avg_Pools = self.Relu(Avg_Pools)
        Avg_Pools = self.conv2(Avg_Pools)
        Out_Feature = torch.multiply(Feature, self.sigmoid(Max_Pools + Avg_Pools))
        ##CBAM_SpatialAttention
        Avg_Pool = torch.mean(Out_Feature, dim=1, keepdim=True)
        Max_Pool, _ = torch.max(Out_Feature, dim=1, keepdim=True)
        Cat_Features = torch.cat([Avg_Pool, Max_Pool], dim=1)
        Feature_New = self.Conv3(Cat_Features)
        Out = torch.multiply(Out_Feature, self.sigmoid2(Feature_New)) + Feature
        Out = Out.view(Features.size(0), -1)
        return Out
if __name__ == "__main__":
    a=1