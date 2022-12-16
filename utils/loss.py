import numpy as np
import numpy.random
import torch
import torch.nn as nn
##设置随机种子，每次训练结果一致
torch.manual_seed(2021)  # 为了使用同样的随机初始化种子以形成相同的随机效果
np.random.seed(2021)
loss_funcMSEL = nn.MSELoss()
loss_CrossEntropy=nn.CrossEntropyLoss()
def creat_lossfunc(Orial_Data,Train_Data,Train_label,Orial_Label):
    Loss_DAE1=loss_funcMSEL(Orial_Data,Train_Data)
    Loss_DAE2=loss_CrossEntropy(Train_label,Orial_Label)
    Loss=Loss_DAE1*0.5+Loss_DAE2*0.5
    return Loss
def L1Regulation(net,lambda1):
    L1_reg = 0
    loss=0
    for param in net.parameters():
        L1_reg += torch.sum(torch.abs(param))
    loss += lambda1 * L1_reg

    return loss






