# MODeepHRD
Homologous recombination deficiency (HRD) is a well-recognized important biomarker in the clinical benefits of platinum-based chemotherapy and PARP inhibitor (PARPi) therapy for patients diagnosed with gynecologic cancers. Accurate prediction of HRD phenotype and simultaneous development of new strategies is still unmet needs and remains challenging. Here, we proposed a novel Multi-Omics integrative Deep-learning framework named MODeepHRD for HRD-positive phenotype detection. The MODeepHRD utilized a convolutional attention autoencoder that can leverage omics-specific and cross-omics complementary knowledge learning. 

The following are the basic applications and operations of our proposed MODeepHRD.
## Requirements
You need to use requirements and download the environment with the same parameters as our library
`pip install -r requirements.txt`

## Data processing
The unified format of our dataset is as follows:
### Expression：
    1.Download the expression profile files from TCGA,quality filtering was performed to exclude low-quality samples with missing more than 20% of the features and features with missing values in more than 20% of the samples;
    2.Screen for signatures of mRNAs, lncRNAs, and miRNAs in expression profiles and propose additional signatures;
    3.Since our network input interface is fixed, construct ExpData according to the Expression in the sample file；
    4.The gene name in each column corresponds to the sample we provided. If the sample does not exist, the gene of the sample needs to be deleted. If the sample exists but the downloaded file does not exist, the gene needs to be filled with 0.
### Methylation：
    1.Download the methylation files from TCGA,quality filtering was performed to exclude low-quality samples with missing more than 20% of the features and features with missing values in more than 20% of the samples;
    2.The gene name in each column corresponds to the sample we provided. If the sample does not exist, the gene of the sample needs to be deleted. If the sample exists but the downloaded file does not exist, the gene needs to be filled with 0.
### Mutation：
    1.Download the somatic mutation files from TCGA,and construct mutation matrix.
    2.Select the genes of all mutation types as the features of the matrix, if the gene mutation in the patient is recorded as 1, otherwise it is recorded as 0.
### Data augmentation：
If you want to get the Methylation or Expression by Generator please run `./datasets/augmentation.py`.
### key code
<pre><code> 
    #Remove columns with missing values greater than 20%
    df = df.dropna(axis=1, thresh=len(df) * 0.8)
    #Remove rows with missing values greater than 20%
    df = df.dropna(axis=0, thresh=len(df.columns) * 0.8)
    #Replace missing values with row mean
    df = df.fillna(df.mean(axis=1))
</code></pre>
## Module Zoo
We provide the .pth file of the optimal model for each module at [Baidu pan:rbdi](https://pan.baidu.com/s/13ptviFu43AEl8t3JJGMSrA?pwd=rbdi)
## Inference
Through the `main.py`, you can start using the modules of MODeepHRD to realize the diagnosis of homologous recombination repair defects in cancer.
Please follow these steps before using in your own dataset. Or use the sample we provided for demonstration.  
+ Setting work mode at `--model`.  
+ Setting datas sample path at `--sample`.  
+ Setting Model parameters at `--config`.  
+ Setting GAN at `--augmentation`.  
+ Run `main.py`.  
When the inference is completed, the results will be stored in `Save_Param`.
## Acknowledgments
+ The implementation of baseline Convolutional Attention Mechanism was based on [cbam](https://github.com/Jongchan/attention-module)


