import numpy as np
import numpy.random
import torch
import torch.utils.data as Data
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score
import math
from model import AutoConvAttention
import argparse
from utils import yaml_config_hook,loss, save_model
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
std=StandardScaler()
##Set a random seed and the results are consistent every time you train
torch.manual_seed(2021)
np.random.seed(2021)
'''
Input data format requirements：
HR_DLabel：The number of labels is consistent with the number of Data, and the samples correspond one-to-one
HR_Files：It is best to use csv format, rows are samples, columns are features, and Value is from 1:n
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    out_dir = args.out_dir
    initial_checkpoint = args.initial_checkpoint
    ##The parameters that affect BATCH are consistent with the space size of the subnetwork layer
    BATCH = args.N ** 2
    ##Number of training sessions per cross-validation
    EPOCH =args.Epoch
    ##The size of the learning rate
    LR = float(args.start_lr)
    ##K-fold cross-validation
    Flod = args.num_fold
    ##Parameters that affect the subspace layer and the prediction layer
    lambda1 = args.lambda1
    ##The final output of the three neural networks is consistent before the operation can be performed
    Out_Feature_size = int(args.Out_Feature_size.split('*')[0])*int(args.Out_Feature_size.split('*')[1])*int(args.Out_Feature_size.split('*')[2])
    ##Read HR files
    HR_DLabel = pd.read_table("../数据/2022_03_01数据/TCGA_OV/Methylaion/Met_Label.csv", sep=',')
    HR_Files = pd.read_csv("../数据/2022_03_01数据/TCGA_OV/Methylaion/Met_Data.csv",sep=',')
    HR_Files.iloc[:, 1:] = min_max_scaler.fit_transform(std.fit_transform(np.array(HR_Files.iloc[:, 1:], dtype=np.float32)))
    ##Tenfold crossover
    Index_HR=np.arange(HR_DLabel.shape[0])
    np.random.shuffle(Index_HR)
    FlodAverage = math.ceil(HR_DLabel.shape[0] / Flod)
    for K in range(0, Flod):
        Index_Test= Index_HR[K*FlodAverage:(K+1)*FlodAverage]
        Index_Train=np.array(list(set(Index_HR).difference(set(Index_Test))),dtype=np.int32)
        np.random.shuffle(Index_Train)
        ##----------------------------------Get a training set------------------------------------##
        Data_Train=np.array(HR_Files.iloc[Index_Train,:].values[:,1:],dtype=np.float32)
        Label_Train=np.array(HR_DLabel.iloc[Index_Train,:].values[:,1:],dtype=np.int64)
        Data_Train_=torch.from_numpy(Data_Train)
        Label_Train_=torch.from_numpy(np.squeeze(Label_Train))
        ##Put the Data and Label of the training set into a Dataframe
        Train_dataset = Data.TensorDataset(Data_Train_, Label_Train_)
        Train_Loder = Data.DataLoader(Train_dataset, batch_size=BATCH, drop_last=False, shuffle=True)
        Feature_Size = Data_Train_.shape[1]
        ##----------------------------------Get the test set------------------------------------##
        Data_Test=HR_Files.iloc[Index_Test,:]
        # Data_Test.iloc[:,1:]=std.transform(np.array(Data_Test.iloc[:,1:],dtype=np.float32))
        Label_Test=HR_DLabel.iloc[Index_Test,:]
        HR_Test=pd.merge(Data_Test,Label_Test,how='left',left_on="Unnamed: 0",
                         right_on="Sample")
        HR_Test.to_csv("Save_Param/Expression_TestData_{}.csv".format(K+1))
        Data_Test_=torch.from_numpy(np.array(Data_Test.values[:,1:],dtype=np.float32))
        Label_Test_=torch.from_numpy(np.array(Label_Test.values[:,1],dtype=np.int64))

        ##Put the Data and Label of the training set into Data
        Linears_F = [Feature_Size ,1024,Out_Feature_size]
        Linears_L = [Out_Feature_size ,1024,Feature_Size]
        def Evaluation_Metrics (y_pre, y_true):
            try:
                AUC = roc_auc_score(y_true, y_pre)
            except ValueError:
                AUC=-1
            ACC=accuracy_score(y_true,y_pre)
            return AUC,ACC


        if initial_checkpoint != 'None':
            f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
            start_epoch = f['epoch']
            start_iteration=f['start_iteration']
            state_dict = f['state_dict']
            AutoConvAttention.Model_.load_state_dict(state_dict, strict=False)  # True
        else:
            start_iteration = 0
            start_epoch = 0
        Model_Deep =AutoConvAttention.Model_(Linears_F,Linears_L,Out_Feature_size)

        optimizer = torch.optim.Adam(Model_Deep.parameters(), lr=LR)
        val_num=0
        for epoch in range(start_epoch,EPOCH):
            Train_num = 0
            Train_loss_epoch=0
            Train_corrects = 0
            Train_AUCsum=0
            for step, (Batch_X, Batch_y) in enumerate(Train_Loder):
                start_iteration+=1
                Train_atten_encoder, Train_decoder,Train_Pre= Model_Deep(Batch_X)

                Train_PreLab = torch.argmax(Train_Pre, 1)
                Train_L1_Loss= loss.L1Regulation(Model_Deep, lambda1)
                Train_Loss= loss.creat_lossfunc(Batch_X, Train_decoder, Train_Pre, Batch_y) + Train_L1_Loss
                optimizer.zero_grad()
                Train_Loss.backward()
                optimizer.step()

                Train_AUCs, _ = Evaluation_Metrics(Train_PreLab.detach().numpy(), Batch_y.detach().numpy())
                Train_AUCsum+=Train_AUCs
                Train_corrects += torch.sum(Train_PreLab == Batch_y.data)
                Train_loss_epoch+=Train_Loss.item()*Batch_X.size(0)
                Train_num=Train_num+Batch_X.size(0)
            ##Calculate the loss for each epoch
            Train_Loss_S=Train_loss_epoch/Train_num
            Train_ACC = Train_corrects.double() / Train_num
            Train_AUC=Train_AUCsum/(step+1)
            ##Test the data for the test
            Test_atten_encoder, Test_decoder, Test_Pre=Model_Deep(Data_Test_)
            Test_Loss = loss.creat_lossfunc(Data_Test_, Test_decoder, Test_Pre, Label_Test_) + Train_L1_Loss
            Test_PreLab = torch.argmax(Test_Pre, 1)
            Test_ACC = torch.sum(Test_PreLab == Label_Test_.data) / Test_PreLab.size(0)
            Test_AUC,_ = Evaluation_Metrics(Test_PreLab.detach().numpy(), Label_Test_.detach().numpy())
            if epoch%10==0:
                print('K :', K, 'Epoch :', epoch, '|',
                      'train loss:%.4f' % Train_Loss_S,
                      'Test loss:%.4f' % Test_Loss.data,
                      'Train|ACC:%.4f'%Train_ACC,
                      'Test|ACC:%.4f' % Test_ACC,
                      'Train|AUC:%.4f' % Train_AUC,
                      'Test|AUC:%.4f' % Test_AUC)
            ##Select the K-fold and need to output the epoch of the model
            # if epochs==Ep[K]:
            #     torch.save(Model_Deep, 'Save_Param/Expression_K{}_Epoch{}.pkl'.format(K, epochs))


