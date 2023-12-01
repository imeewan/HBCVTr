#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BartTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import PreTrainedTokenizer


class CustomBartModel(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        last_encoder_output = outputs.encoder_last_hidden_state
        return last_encoder_output    
    


# In[ ]:




