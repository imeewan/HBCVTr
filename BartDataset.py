import torch
from torch.utils.data import Dataset


class BartDataset(Dataset):
    def __init__(self, input_encodings, labels):
        self.input_encodings = input_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.input_encodings['input_ids'])
