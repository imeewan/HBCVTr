#!/usr/bin/env python
# coding: utf-8

import json
import os

from BartDataset import BartDataset
from CustomBart_Atomic_Tokenizer import CustomBart_Atomic_Tokenizer
from CustomBart_FG_Tokenizer import CustomBart_FG_Tokenizer
from DualInputDataset import DualInputDataset
from transformers import BartForConditionalGeneration, BartConfig, PreTrainedTokenizer
from rdkit import Chem
from rdkit.Chem import SaltRemover

remover = SaltRemover.SaltRemover()

def remove_salt(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = remover.StripMol(mol)
        smiles = Chem.MolToSmiles(mol)
    return smiles

_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
with open(os.path.join(_data_dir, 'atomic_vocab.json')) as f:
    atomic_vocab = json.load(f)
with open(os.path.join(_data_dir, 'fg_vocab.json')) as f:
    fg_vocab = json.load(f)

tokenizer1 = CustomBart_Atomic_Tokenizer(vocab=atomic_vocab)
tokenizer1.pad_token = '_'
tokenizer1.pad_token_id = tokenizer1.convert_tokens_to_ids(tokenizer1.pad_token)

tokenizer2 = CustomBart_FG_Tokenizer(vocab=fg_vocab)
tokenizer2.pad_token = '_'
tokenizer2.pad_token_id = tokenizer2.convert_tokens_to_ids(tokenizer2.pad_token)

max_length = 250
batch_size = 64

config1 = BartConfig(
    vocab_size=len(atomic_vocab),
    d_model=512,
    encoder_ffn_dim=4096,
    num_attention_heads=64,
    num_hidden_layers=6,
    pad_token_id=tokenizer1.pad_token_id,
    max_position_embeddings=max_length,
    dropout=0,
)

config2 = BartConfig(
    vocab_size=len(fg_vocab),
    d_model=512,
    encoder_ffn_dim=4096,
    num_attention_heads=64,
    num_hidden_layers=6,
    pad_token_id=tokenizer2.pad_token_id,
    max_position_embeddings=max_length,
    dropout=0,
)

reg_mod = [1024, 640]

max_pact_hcv = 11
min_pact_hcv = 3.37
max_pact_hbv = 10.22
min_pact_hbv = 4
