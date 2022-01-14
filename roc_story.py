import csv
import torch
import gensim.downloader
from torch.utils.data import Dataset
from transformers.models.bert.tokenization_bert import BertTokenizer
from torch.nn.utils.rnn import pack_padded_sequence
from transformers.optimization import get_linear_schedule_with_warmup


class ROCDataset(Dataset):
    def __init__(self, text_dir):
        self.data = list()
        with open(text_dir) as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append([row[1:7], int(row[7]) - 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx][0]
        label = self.data[idx][1]
        return words, label
