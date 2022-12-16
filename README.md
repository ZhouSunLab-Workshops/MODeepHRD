# MODeepHRD
Homologous recombination repair (HRR) is a DNA repair mechanism in which damaged DNA fragments are repaired by reference sequences that are similar to other sequences. This repair mechanism plays an important role in cells because it helps cells repair DNA damage caused by external factors such as free radicals and UV radiation.

However, when HRR is defective, cells may be unable to repair DNA damage, which can lead to cancer. Therefore, predicting HRR deficiency may have important implications for early diagnosis and treatment of cancer.

In the past few years, deep learning (deep learning) has achieved great success in the field of medicine and has played an important role in cancer diagnosis and prediction. Therefore, many researchers try to predict HRR deficits using deep learning.

For example, studies have used deep learning algorithms to analyze genomic data to predict the presence of HRR defects in cancer cells. These studies demonstrate that deep learning can effectively predict HRR deficits with high accuracy.

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
### Data augmentation
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
We provide the .pth file of the optimal model for each module at [Baidu pan]()..
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
+ The implementation of baseline Convolutional Attention Mechanism was based on [timm]([https://github.com/rwightman/pytorch-image-models#introduction](https://github.com/bruinxiong/Modified-CBAMnet.mxnet)). 


