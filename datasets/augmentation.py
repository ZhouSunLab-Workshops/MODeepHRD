import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
torch.manual_seed(2022)
np.random.seed(2022)
std=StandardScaler()
from tensorboardX import SummaryWriter
class GAN_Discriminator(nn.Module):
    def __init__(self,Feature_Nums):
        super(GAN_Discriminator, self).__init__()
        self.Discriminator=nn.Sequential(
            nn.Linear(Feature_Nums, 256),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(256),
            ##Broaden deep network
            nn.Linear(256, 256),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.Linear(256, 16),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(16),
            ##输出判别器状态量
            nn.Linear(16, 1),
            nn.Sigmoid()

        )
        self.loss_function = nn.BCELoss()
        self.Discriminator_optimiser=torch.optim.Adam(self.parameters(), lr=0.001)
    def forward(self,inputs):
        D_Label = self.Discriminator(inputs)
        return D_Label
    def train(self,inpit,target):
        Label=self.Discriminator(inpit)
        self.loss=self.loss_function(Label,target)
        self.Discriminator_optimiser.zero_grad()
        self.loss.backward()
        self.Discriminator_optimiser.step()

        pass


class GAN_Generator(nn.Module):
    def __init__(self,Feature_Idears,Feature_Nums):
        super(GAN_Generator, self).__init__()

        self.Generator=nn.Sequential(
            nn.Linear(Feature_Idears, 1024),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(1024),
            ##Broaden deep network
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(1024),
            ##Output generator expressions
            nn.Linear(1024, Feature_Nums),
            nn.Sigmoid()
        )
        self.count = 0
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)
    def forward(self, idears):
        Generate_Data = self.Generator(idears)

        return Generate_Data
    def train(self,Discriminator,idears,D_label):
        Generate_Data=self.forward(idears)
        G_Label=Discriminator.forward(Generate_Data)

        self.loss=Discriminator.loss_function(G_Label,D_label)
        print("G Loss%.4f" % self.loss.item())
        self.optimiser.zero_grad()
        self.loss.backward()
        self.optimiser.step()
        self.count+=1
        pass

def generate_random(size):
    random_data=torch.randn(size)
    return random_data
def Data_Choice(Samples):

    Data_Positive_Data =pd.read_csv("YourDatasets",sep=',')
    Data_Positive_Label= pd.read_csv("YourDatasets_labels",sep=',')
    Data_Positive_Data_=Data_Positive_Data.iloc[list(Data_Positive_Label.values[:,1]==Samples),:]
    Data_Positive_value=Data_Positive_Data_.values[:,1:]
    return Data_Positive_value

if __name__ == '__main__':
    writer = SummaryWriter('log')
    ##Generate 0 label samples /Generate 1 label sample
    Data_Positive_value=Data_Choice(0)

    Data_Positive_value=std.fit_transform(Data_Positive_value)
    Data_Positive_value=torch.from_numpy(np.array(Data_Positive_value,dtype=np.float32))
    ##There are 35,737 genes
    Max_Sample_Num=256
    Discriminator=GAN_Discriminator(Data_Positive_value.shape[1])
    Generator=GAN_Generator(Max_Sample_Num,Data_Positive_value.shape[1])
    Index=1
    for epoch in range(1000):
        for indexs,Val_Feature in enumerate(Data_Positive_value):
            Discriminator.train(Val_Feature,torch.FloatTensor([1.0]))
            Discriminator.train(Generator(generate_random(Max_Sample_Num)).detach(),torch.FloatTensor([0.0]))
            Generator.train(Discriminator,generate_random(Max_Sample_Num),torch.FloatTensor([1.0]))
            writer.add_scalar('discriminators/Loss', Discriminator.loss.data.numpy(), Index)
            writer.add_scalar('Generators/Loss', Generator.loss.data.numpy(),Index)
            Index+=1
            if Index%50==0:
                print("Epoch:",epoch," Index:",Index)
    torch.save(Generator, 'Model/Methylation_OR_Expression.pkl')







