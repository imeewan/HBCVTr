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
from torch.utils.data import Dataset, DataLoader
from transformers import BartConfig, AdamW, get_linear_schedule_with_warmup

class CustomBartModel(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        last_encoder_output = outputs.encoder_last_hidden_state
        return last_encoder_output    

class DualBartModel(nn.Module):
    def __init__(self, config1, config2, hidden_sizes):
        super().__init__()
        self.bart1 = CustomBartModel(config1)
        self.bart2 = CustomBartModel(config2)

        self.ffn = nn.ModuleList()

        # Create the hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.ffn.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.ffn.append(nn.ReLU())

        self.regression_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        output1 = self.bart1(input_ids1, attention_mask1)
        output2 = self.bart2(input_ids2, attention_mask2)
        merged = torch.cat((output1, output2), dim=-1)
        pooled = merged.mean(dim=1)  # Mean pooling

        # Pass through the hidden layers
        for layer in self.ffn:
            pooled = layer(pooled)

        pred = self.regression_head(pooled)
        return pred.squeeze(-1)  # to get shape [batch_size]


# In[ ]:




