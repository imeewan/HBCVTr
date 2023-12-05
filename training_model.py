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
from utils import *



if __name__ == "__main__":

    max_length = 250
    batch_size = 8

    data_path = "data/hbv_dataset.csv"
    train_dataloader, val_dataloader = train_val_proc(data_path)

    d_models = [128]
    encoder_ffn_dims = [256]
    num_attention_heads = [8]
    num_hidden_layers = [2]
    dropouts = [0.15]
    learning_rates = [1e-6] 
    reg_mod = [256, 128]
    weight_decay = 0.001
    num_epochs = 2

    param_combinations = list(itertools.product(d_models, encoder_ffn_dims, num_attention_heads, num_hidden_layers, dropouts, learning_rates))
    combinations = param_combinations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                torch.save(model.state_dict(), f"model/new_model.pt")
                max_r2 = avg_val_r2

                

# In[3]:




