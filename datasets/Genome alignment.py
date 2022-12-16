import numpy as np
import pandas as pd
import os
class Express_replace():
    def __init__(self,Data_1,Data_2,Data_3name):
        super(Express_replace, self).__init__()
        '''
        Align the gene format of Data1 to the gene format of Data2
        Data1 and Data2 are both behavioral samples listed as gene names
        eg：
            Data1：100x200
            Data2：150x300
            Output:DATA3:100X300
        '''
        ##Get the gene names of Data1 and Data2
        Data_1_GeneName=Data_1.axes[1].values[1:]
        Data_2_GeneName=Data_2.axes[1].values[1:]
        ##Get the Values of Data1 and add the last column of Value as column 0, so that the gene of Data2 does not exist in Data1, replace it with column 0
        Data_1_=Data_1.values[:,1:]
        Ze=np.zeros((Data_1_.shape[0],1),dtype=np.float32)
        New_OV_Data_1_Value=np.hstack((Data_1_,Ze))
        ##Define a new matrix with the same Data2 gene format
        New_Marit = np.zeros((Data_1_.shape[0], Data_2.shape[1]-1),dtype=np.float32)
        ##Populate the new matrix
        i=0
        for GeneName in Data_2_GeneName:
            Index_=np.squeeze(np.argwhere(GeneName==Data_1_GeneName))
            if(Index_.size!=0):
                New_Marit[:,i]=New_OV_Data_1_Value[:,Index_]
            else:
                New_Marit[:, i]=New_OV_Data_1_Value[:,-1]
            i+=1
        ##Construct the new file
        Sample_Name=Data_1.values[:,0]
        Gene_Name=Data_2_GeneName
        Data_1_Values=pd.DataFrame(New_Marit,columns=Gene_Name,index=Sample_Name)
        Data_1_Values.to_csv(Data_3name)
if __name__ == "__main__":
    Data_2 = pd.read_table("Exp_Data.csv", sep=',')
    In_Put="Expression/"
    Out_Put="Exp_NewFiles/"
    File_Input=os.listdir(In_Put)
    for files in File_Input:
        File_1=pd.read_csv(In_Put+files,sep=',')
        Express_replace(File_1, Data_2, Out_Put+files.split('.')[0]+'.csv')
        print("succeed！")
