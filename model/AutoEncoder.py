import math
import numpy as np
import numpy.random
import torch
import torch.nn as nn
import ConvAttention
torch.manual_seed(2021)  # To initialize the seed with the same randomness to form the same random effect
np.random.seed(2021)

class AttenDAE(nn.Module):
    def __init__(self,Linears_F,Linears_L):
        super(AttenDAE, self).__init__()
        self.DAE_Model_Encoder=self.create_Enconder_layers(Linears_F)
        self.DAE_Model_Decoder=self.create_Deconder_layers(Linears_L)
        self.CBAM_Model= ConvAttention.CBAM(in_channels=3, out_channels=16, Feature_Num=Linears_F[-1], Out_Feature_size=Linears_F[-1])

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
if __name__ == "__main__":
    a=1