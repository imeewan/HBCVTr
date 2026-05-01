![image](https://github.com/imeewan/HBCVTr/assets/29390962/b512b33e-227d-4252-b9aa-6267ad9dea6d)

# HBCVTr

HBCVTr is a double-encoder transformer and deep neural network machine learning model to predict the antiviral activity against hepatitis B virus (HBV) and hepatitis C virus (HCV) using a simplified molecular-input line-entry system (SMILES) of small molecules.

**Publication:** [*Scientific Reports* (2024)](https://www.nature.com/articles/s41598-024-59933-4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imeewan/HBCVTr/blob/main/HBCVTr_Prediction_Demo.ipynb)

## Requirements

| Package | Version |
|---|---|
| Python | 3.11.4 |
| numpy | 1.25.0 |
| pandas | 1.5.3 |
| torch | 2.0.1 |
| rdkit | 2023.3.2 |
| tqdm | 4.65.0 |
| transformers | 4.31.0 |
| scikit-learn | 1.2.2 |
| deepsmiles | 1.0.1 |
| SmilesPE | 0.0.3 |

## Quick Start (Google Colab)

Click the badge above or [open the demo notebook in Colab](https://colab.research.google.com/github/imeewan/HBCVTr/blob/main/HBCVTr_Prediction_Demo.ipynb). No local installation required — just run the four steps in the notebook.

## Local Installation

### 1. Create the conda environment

```bash
conda create -c conda-forge -n hbcv rdkit=2023.3.2 -y
conda activate hbcv
conda install numpy=1.25.0 pandas=1.5.3 scikit-learn=1.2.2 tqdm=4.65.0 pytorch=2.0.1 -c pytorch -y
pip install transformers==4.31.0 SmilesPE==0.0.3 deepsmiles
```

### 2. Download the trained models

Download `hbv_model.pt` and `hcv_model.pt` from [Google Drive](https://drive.google.com/drive/folders/1yRFQs9Hl8AfA3f-GvsnP7w-0oionkBaU?usp=sharing) and place them in the `model/` directory.

### 3. Run prediction

```bash
python predict.py
```

Enter your SMILES and choose the target virus:

```
Enter the SMILES of the compound: C[C@H](Cn1cnc2c(N)ncnc21)OCP(=O)(O)OP(=O)(O)CO[C@H](C)Cn1cnc2c(N)ncnc21
Predict activity against HBV or HCV? (Enter HBV or HCV): HCV
```

Example output:

```
SMILES:           C[C@H](Cn1cnc2c(N)ncnc21)OCP(=O)(O)OP(=O)(O)CO[C@H](C)Cn1cnc2c(N)ncnc21
Predicted pACT:   8.1230
Predicted EC50:   7.5343 nM
```

## Repository Structure

```
HBCVTr/
├── data/
│   ├── atomic_vocab.txt           # Atom-level tokeniser vocabulary
│   ├── fg_vocab.txt               # Functional-group tokeniser vocabulary
│   ├── spe_vocab_list.txt         # SPE merge operations
│   ├── hbv_dataset.csv            # HBV training dataset
│   └── hcv_dataset.csv            # HCV training dataset
├── model/
│   └── dummy_model.pt             # Placeholder (download real models from Google Drive)
├── BartDataset.py                 # Dataset class
├── CustomBart_Atomic_Tokenizer.py # Atom-level tokeniser
├── CustomBart_FG_Tokenizer.py     # Functional-group tokeniser
├── CustomBartModel.py             # Single BART encoder wrapper
├── DualBartModel.py               # Dual-encoder + regression head
├── DualInputDataset.py            # Dual-input dataset
├── TqdmWrap.py                    # Progress-bar wrapper
├── pretrained_utils.py            # Tokenisers and model config for inference
├── predict.py                     # Command-line prediction script
├── training_model.py              # Model training script
├── utils.py                       # Training utilities
├── environment.yml                # Conda environment specification
└── HBCVTr_Prediction_Demo.ipynb   # Interactive Colab demo notebook
```
