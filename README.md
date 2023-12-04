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
    <html>
      <head>
      </head>
    </html>
