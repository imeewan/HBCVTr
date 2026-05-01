from transformers import BartForConditionalGeneration


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
