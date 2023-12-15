# HBCVTr
HBCVTr is double-encoder of transformers and deep neural network machine learning model to predict the antiviral activity against hepatitis B virus (HBV) and hepatitis C virus (HCV) using a simplified molecular-input line-entry system (SMILES) of small molecules 
# Requirement
python: 3.11.4\
numpy: 1.25.0\
pandas: 1.5.3\
torch: 2.0.1\
rdkit: 2023.3.2\
tqdm: 4.65.0\
transformers: 4.31.0\
scikit-learn: 1.2.2\
deepsmiles: 1.0.1\
SmilesPE: 0.0.3

# Installing dependencies

To run HBCVTr on your local machine, we recommend using Anaconda as the package control. You have to install the dependencies first.  

## Using CPU
If your local machine does not have a CUDA-enabled GPU, you can install the packages with CPU support. 

To create conda environment and install the depedencies, execute the following commands:

        conda create -c conda-forge -n hbcv rdkit -y
        conda activate hbcv
        conda install numpy=1.25.0 pandas=1.5.3 scikit-learn=1.2.2 tqdm=4.65.0 pytorch=2.0.1 -c pytorch -y
        pip install transformers==4.31.0 SmilesPE==0.0.3
        pip install --upgrade deepsmiles
        
Ensure that you activate 'hbcv' environment before installing these packages

## Using GPU
To use GPU to accelerate computation, first install CUDA toolkit and CUDNN before installing Pytorch.

        conda create -c conda-forge -n hbcv rdkit -y
        conda activate hbcv
        conda install numpy=1.25.0 pandas=1.5.3 scikit-learn=1.2.2 tqdm=4.65.0
        conda install -c anaconda cudatoolkit
        conda install -c anaconda cudnn
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
        pip install transformers==4.31.0 SmilesPE==0.0.3
        pip install --upgrade deepsmiles

# Trained models
The trained models for biological activity prediction against HBV and HCV are available at:/
https://drive.google.com/drive/folders/1yRFQs9Hl8AfA3f-GvsnP7w-0oionkBaU?usp=sharing

# Performing the prediction
Make sure that the trained model is downloaded and stored in the model folder. Then, execute predict.py
Enter your SMILES

    Enter the SMILES of the compound: C[C@H](Cn1cnc2c(N)ncnc21)OCP(=O)(O)OP(=O)(O)CO[C@H](C)Cn1cnc2c(N)ncnc21

Select whether you want to predict the compound's activity against HBV or HCV

    Do you want to predict the compound's activity against HBV or HCV? (Enter HBV or HCV): HCV

Then the prediction results will show up
    
    SMILES:  C[C@H](Cn1cnc2c(N)ncnc21)OCP(=O)(O)OP(=O)(O)CO[C@H](C)Cn1cnc2c(N)ncnc21
    Predicted pACT:  8.12295828104019
    Predicted EC50 : 7.534279356412855 nM
    
# Using Google Colab

An alternative to using your local machine is to use Google Colab to explore HBCVTr. Google Colab will provide an easy way to quickly test the prediction and to train the models. 

The Python notebook (HBCVTr_Demo.ipynb file) is available in the folder Colab. Upload the file to your Google Colab and then run the script to explore HBCVTr. The notebook contains the detailed explanation of how to run the script on Google Colab.