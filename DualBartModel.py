import torch
from torch import nn
from transformers import BartForConditionalGeneration, BartConfig


class CustomBartModel(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.encoder_last_hidden_state


class DualBartModel(nn.Module):
    def __init__(self, config1, config2, hidden_sizes):
        super().__init__()
        self.bart1 = CustomBartModel(config1)
        self.bart2 = CustomBartModel(config2)

        self.ffn = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.ffn.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.ffn.append(nn.ReLU())

        self.regression_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        output1 = self.bart1(input_ids1, attention_mask1)
        output2 = self.bart2(input_ids2, attention_mask2)
        merged = torch.cat((output1, output2), dim=-1)
        pooled = merged.mean(dim=1)

        for layer in self.ffn:
            pooled = layer(pooled)

        pred = self.regression_head(pooled)
        return pred.squeeze(-1)
