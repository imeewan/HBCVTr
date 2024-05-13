![image](https://github.com/imeewan/HBCVTr/assets/29390962/b512b33e-227d-4252-b9aa-6267ad9dea6d)

# HBCVTr
HBCVTr is double-encoder of transformers and deep neural network machine learning model to predict the antiviral activity against hepatitis B virus (HBV) and hepatitis C virus (HCV) using a simplified molecular-input line-entry system (SMILES) of small molecules. The article has been published at: https://www.nature.com/articles/s41598-024-59933-4
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
To create conda environment and innstall the depedencies, execute the following commands:

        conda create -c conda-forge -n hbcv rdkit -y
        conda activate hbcv
        conda install numpy=1.25.0 pandas=1.5.3 scikit-learn=1.2.2 tqdm=4.65.0 pytorch=2.0.1 -c pytorch -y
        pip install transformers==4.31.0 SmilesPE==0.0.3
        pip install --upgrade deepsmiles
        
Ensure that you activate 'hbcv' environment before installing these packages

# Trained models
The trained models for biological activity prediction against HBV and HCV are available at:/
https://drive.google.com/drive/folders/1yRFQs9Hl8AfA3f-GvsnP7w-0oionkBaU?usp=sharing

# Performing the prediction
Execute predict.py
Enter your SMILES

    Enter the SMILES of the compound: C[C@H](Cn1cnc2c(N)ncnc21)OCP(=O)(O)OP(=O)(O)CO[C@H](C)Cn1cnc2c(N)ncnc21

Then select whether you want to predict the compound's activity against HBV or HCV

    Do you want to predict the compound's activity against HBV or HCV? (Enter HBV or HCV): HCV

The the prediction results will show up
    
    SMILES:  C[C@H](Cn1cnc2c(N)ncnc21)OCP(=O)(O)OP(=O)(O)CO[C@H](C)Cn1cnc2c(N)ncnc21
    Predicted pACT:  8.12295828104019
    Predicted EC50 : 7.534279356412855 nM
    
