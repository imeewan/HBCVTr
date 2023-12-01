#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

def tokenize_with_progress(tokenizer, smiles_list, **kwargs):
    tokenized_smiles = []
    for smiles in tqdm(smiles_list, desc="Tokenizing"):
        pass
        try:
            tokenized_smiles.append(tokenizer(smiles, **kwargs))
        except Exception as e: 
            pass
    return tokenized_smiles

def batch_encode_with_progress(tokenizer, smiles_list, **kwargs):
    tokenized_smiles = []
    for smiles in tqdm(smiles_list, desc="Tokenizing"):
        pass

        try:
            tokenized_smiles.append(tokenizer(smiles, **kwargs))
        except Exception as e:
            pass
            raise e 
    return tokenizer.pad(tokenized_smiles, padding='max_length', max_length=max_length, return_tensors='pt')

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
            
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
            
            
def dataprep(text):
    smiles = text['smiles'].tolist()
    labels = text['pACT'].tolist()
    labels_array = np.array(labels)
    
    return smiles, labels_array

def atomic_voc_load():
    file_path = 'data/atomic_vocab.txt'
    atomic_vocab = []

    with open(file_path, 'r') as file:
        for line in file:
            for item in line.split(','):
                atomic_vocab.append(item.strip().strip("'").strip('"'))
    return atomic_vocab

def fg_voc_load():
    file_path = 'data/fg_vocab.txt'
    fg_vocab = []

    with open(file_path, 'r') as file:
        for line in file:
            for item in line.split(','):
                fg_vocab.append(item.strip().strip("'").strip('"'))

    return fg_vocab

def norm_label(labels_array):
    min_val = np.min(labels_array)
    max_val = np.max(labels_array)
    normalized_labels = (labels_array - min_val) / (max_val - min_val)
    labels = normalized_labels.tolist()
    return labels

def data_preproc(smiles_data, labels):

    train_smiles, val_smiles, train_labels, val_labels = train_test_split(smiles_data, labels, test_size=0.2, random_state=47)
    input_encodings1_train = batch_encode_with_progress(tokenizer1, train_smiles, truncation=True, max_length=max_length, padding='max_length')
    input_encodings1_val = batch_encode_with_progress(tokenizer1, val_smiles, truncation=True, max_length=max_length, padding='max_length')

    input_encodings2_train = batch_encode_with_progress(tokenizer2, train_smiles, truncation=True, max_length=max_length, padding='max_length')
    input_encodings2_val = batch_encode_with_progress(tokenizer2, val_smiles, truncation=True, max_length=max_length, padding='max_length')

    train_dataset = DualInputDataset(input_encodings1_train['input_ids'], 
                                     input_encodings1_train['attention_mask'],
                                     input_encodings2_train['input_ids'], 
                                     input_encodings2_train['attention_mask'],
                                     train_labels)
    val_dataset = DualInputDataset(input_encodings1_val['input_ids'], 
                                   input_encodings1_val['attention_mask'],
                                   input_encodings2_val['input_ids'], 
                                   input_encodings2_val['attention_mask'],
                                   val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader

def train_val_proc(data_path):
    df = pd.read_csv(data_path)
    smiles_data, labels = dataprep(df)

    labels = norm_label(labels)

    train_dataloader, val_dataloader = data_preproc(smiles_data, labels)
    
    return train_dataloader, val_dataloader

def training_model(combinations):
    for idx, (d_model1, encoder_ffn_dim1, num_attention_heads1, num_hidden_layers1, dropout1, lr1) in enumerate(combinations):

        (d_model2, encoder_ffn_dim2, num_attention_heads2, num_hidden_layers2, dropout2, lr2) = (d_model1, encoder_ffn_dim1, num_attention_heads1, num_hidden_layers1, dropout1, lr1)

        max_r2 = -100

        config1 = BartConfig(
            vocab_size=len(atomic_vocab),
            d_model=d_model1,
            encoder_ffn_dim=encoder_ffn_dim1,
            num_attention_heads=num_attention_heads1,
            num_hidden_layers=num_hidden_layers1,
            pad_token_id=tokenizer1.pad_token_id,
            max_position_embeddings=max_length,
            dropout=dropout1,
        )

        config2 = BartConfig(
            vocab_size=len(fg_vocab),
            d_model=d_model2,
            encoder_ffn_dim=encoder_ffn_dim2,
            num_attention_heads=num_attention_heads2,
            num_hidden_layers=num_hidden_layers2,
            pad_token_id=tokenizer2.pad_token_id,
            max_position_embeddings=max_length,
            dropout=dropout2,
        )


        model = DualBartModel(config1, config2, reg_mod)
        model.to(device)
        model.apply(weights_init)
        optimizer = AdamW(model.parameters(), lr=lr1, weight_decay=weight_decay)

        print(f"Model {idx+1} configurations: ")
        print(f"d_model1: {d_model1}, encoder_ffn_dim1: {encoder_ffn_dim1}, num_attention_heads1: {num_attention_heads1}, num_hidden_layers1: {num_hidden_layers1}")
        print(f"d_model2: {d_model2}, encoder_ffn_dim2: {encoder_ffn_dim2}, num_attention_heads2: {num_attention_heads2}, num_hidden_layers2: {num_hidden_layers2}")

        log_file_path = f"model/hcv_model.log"

        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        config_dict = {
            'd_model1': d_model1,
            'encoder_ffn_dim1': encoder_ffn_dim1,
            'num_attention_heads1': num_attention_heads1,
            'num_hidden_layers1': num_hidden_layers1,
            'd_model2': d_model2,
            'encoder_ffn_dim2': encoder_ffn_dim2,
            'num_attention_heads2': num_attention_heads2,
            'num_hidden_layers2': num_hidden_layers2,
            'dropout1': dropout1,
            'dropout2': dropout2,
            'lr': lr1,
            'regression_dim': reg_mod,
            'weight_decay': weight_decay,
        }

        with open(log_file_path, 'w') as outfile:
            outfile.write(json.dumps(config_dict) + '\n')

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True)):
                inputs1, inputs2 = batch['input_ids1'].to(device).long(), batch['input_ids2'].to(device).long()
                attention_mask1, attention_mask2 = batch['attention_mask1'].to(device).long(), batch['attention_mask2'].to(device).long()

                labels = batch['labels'].to(device).float()
                optimizer.zero_grad()

                outputs = model(input_ids1=inputs1, attention_mask1=attention_mask1,
                                input_ids2=inputs2, attention_mask2=attention_mask2)
                pred = outputs
                loss = torch.nn.MSELoss()(pred, labels)

                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_dataloader)            

            model.eval()
            total_eval_loss = 0
            total_eval_r2 = 0

            # Evaluate data for one epoch
            for batch in val_dataloader:
                with torch.no_grad():
                    inputs1, inputs2 = batch['input_ids1'].to(device).long(), batch['input_ids2'].to(device).long()
                    attention_mask1, attention_mask2 = batch['attention_mask1'].to(device).long(), batch['attention_mask2'].to(device).long()

                    labels = batch['labels'].to(device).float()

                    outputs = model(input_ids1=inputs1, attention_mask1=attention_mask1,
                                    input_ids2=inputs2, attention_mask2=attention_mask2)
                    pred = outputs
                    loss = torch.nn.MSELoss()(pred, labels)
                    total_eval_loss += loss.item()
                    total_eval_r2 += r2_score(labels.cpu().numpy(), pred.cpu().detach().numpy())

            avg_val_loss = total_eval_loss / len(val_dataloader)
            avg_val_r2 = total_eval_r2 / len(val_dataloader)

            log_dict = {
                'epoch': epoch+1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'avg_val_r2': avg_val_r2,

            }

            with open(log_file_path, 'a') as outfile:
                outfile.write(json.dumps(log_dict) + '\n')

            if avg_val_r2 > max_r2:
                torch.save(model.state_dict(), f"model/hcv_model.pt")
                max_r2 = avg_val_r2
                print(max_r2)
                
atomic_vocab = atomic_voc_load()
fg_vocab = fg_voc_load()
tokenizer1 = CustomBart_Atomic_Tokenizer(vocab=atomic_vocab)
tokenizer1.pad_token = '_'
tokenizer1.pad_token_id = tokenizer1.convert_tokens_to_ids(tokenizer1.pad_token)

tokenizer2 = CustomBart_FG_Tokenizer(vocab=fg_vocab)
tokenizer2.pad_token = '_'
tokenizer2.pad_token_id = tokenizer2.convert_tokens_to_ids(tokenizer2.pad_token)

max_length = 250
batch_size = 64


# In[ ]:




