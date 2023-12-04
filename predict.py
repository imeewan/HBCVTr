#!/usr/bin/env python
# coding: utf-8

# In[15]:


from BartDataset import BartDataset
from CustomBart_Atomic_Tokenizer import CustomBart_Atomic_Tokenizer
from CustomBart_FG_Tokenizer import CustomBart_FG_Tokenizer
from TqdmWrap import TqdmWrap
from DualInputDataset import DualInputDataset
from DualBartModel import DualBartModel, CustomBartModel
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.optim import AdamW
import pandas as pd
import numpy as np
import random
import deepsmiles
from SmilesPE.tokenizer import *
from SmilesPE.pretokenizer import atomwise_tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
import codecs
from transformers import AdamW, BartTokenizer, BartForConditionalGeneration, BartConfig, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizer
import re
from tqdm.auto import tqdm
from tqdm import tqdm
import itertools
import json
import os
from utils import *
from pretrained_utils import *
from rdkit import Chem
from rdkit.Chem import SaltRemover


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smiles = input("Enter the SMILES of the compound: ")

    virus_choice = input("Do you want to predict the compound's activity against HBV or HCV? (Enter HBV or HCV): ").lower()

    print("Analysis in progress ...")

    if virus_choice == 'hbv':
        model_path = "model/hbv_model.pt"
        max_pact = max_pact_hbv
        min_pact = min_pact_hbv
    elif virus_choice == 'hcv':
        model_path = "model/hcv_model.pt"
        max_pact = max_pact_hcv
        min_pact = min_pact_hcv
    else:
        raise ValueError("Invalid input. Please enter either 'HBV' or 'HCV'.")

    max_length = 250
    
    model = DualBartModel(config1, config2, reg_mod)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    smiles_data_no_salt = remove_salt(smiles)
    smiles = smiles_data_no_salt

    input_encoding1 = tokenizer1.encode_plus(smiles, truncation=True, max_length=max_length, padding='max_length', return_tensors="pt")
    input_encoding2 = tokenizer2.encode_plus(smiles, truncation=True, max_length=max_length, padding='max_length', return_tensors="pt")

    input_ids1 = input_encoding1['input_ids'].to(device)
    attention_mask1 = input_encoding1['attention_mask'].to(device)
    input_ids2 = input_encoding2['input_ids'].to(device)
    attention_mask2 = input_encoding2['attention_mask'].to(device)
    

    with torch.no_grad():
        output = model(input_ids1=input_ids1, attention_mask1=attention_mask1,
                       input_ids2=input_ids2, attention_mask2=attention_mask2)
    
    prediction = output
    prediction_value = prediction.cpu().numpy()[0]
    print('SMILES: ', smiles)
    print('Predicted pACT: ', prediction_value * (max_pact - min_pact) + min_pact)
    predicted_EC50 = 10**-(prediction_value * (max_pact - min_pact) + min_pact) * 10**9
    print('Predicted EC50 :', predicted_EC50, 'nM')


# In[ ]:




