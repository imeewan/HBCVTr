#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import BartTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import PreTrainedTokenizer
from transformers import BartConfig, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

class BartDataset(torch.utils.data.Dataset):
    def __init__(self, input_encodings, labels):
        self.input_encodings = input_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.input_encodings['input_ids'])


# In[ ]:




