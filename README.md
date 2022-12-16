# MODeepHRD
Prediction of homologous recombination and repair defects based on omics data
## Requirements
You need to use requirements and download the environment with the same parameters as our library
pip install -r packages.txt

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
  1.Download the somatic mutation files from TCGA,


