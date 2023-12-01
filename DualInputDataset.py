#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset, DataLoader

class DualInputDataset(Dataset):
    def __init__(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels):
        self.input_ids1 = input_ids1
        self.attention_mask1 = attention_mask1
        self.input_ids2 = input_ids2
        self.attention_mask2 = attention_mask2
        self.labels = labels

    def __len__(self):
        return len(self.input_ids1)

    def __getitem__(self, idx):
        return {
            'input_ids1': self.input_ids1[idx],
            'attention_mask1': self.attention_mask1[idx],
            'input_ids2': self.input_ids2[idx],
            'attention_mask2': self.attention_mask2[idx],
            'labels': self.labels[idx]
        }


# In[ ]:




