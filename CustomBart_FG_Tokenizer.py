import codecs
import deepsmiles
from SmilesPE.tokenizer import SPE_Tokenizer
from transformers import PreTrainedTokenizer


class CustomBart_FG_Tokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.converter = deepsmiles.Converter(rings=True, branches=True)
        self.encoder = {char: idx for idx, char in enumerate(vocab)}
        self.decoder = {idx: char for idx, char in enumerate(vocab)}
        self.spe_vob = codecs.open('data/spe_vocab_list.txt')

    def tokenize(self, text):
        spe = SPE_Tokenizer(self.spe_vob)
        tokens = spe.tokenize(text)
        return tokens.split(' ')

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.encoder.get(tokens, self.encoder.get('[UNK]'))
        return [self.encoder.get(t, self.encoder.get('[UNK]')) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.decoder.get(ids, '[UNK]')
        return [self.decoder.get(i, '[UNK]') for i in ids]

    @property
    def pad_token_id(self):
        return self.encoder.get(self.pad_token, self.encoder.get('[UNK]'))

    @pad_token_id.setter
    def pad_token_id(self, value):
        self.pad_token = self.convert_ids_to_tokens(value)
