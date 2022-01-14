import torch
import os
import sys
import pandas as pd

class WeeBitDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        self.dirs = os.listdir(datapath)
        self.data_dict = {'WRLevel2': [], 'WRLevel4': [], 'WRLevel3': [], 'BitGCSE': [], 'BitKS3': []}
        for dir in self.dirs:
            if dir != '.DS_Store':
                files = os.listdir('WeeBit-TextOnly/' + dir)
                for file in files:
                    if file != '.DS_Store':
                        f = open('WeeBit-TextOnly/' + dir + '/' + file, 'rb')
                        text = f.read()
                        text = text.decode('utf-8', errors='ignore').replace('\n', ' ')
                        self.data_dict[dir].append(text)
                        f.close()
        self.text_to_label = {'WRLevel2': 0, 'WRLevel3': 1, 'WRLevel4': 2, 'BitKS3': 3, 'BitGCSE': 4}
        self.df = pd.DataFrame(columns=['text', 'label'])
        count = 0
        for key in list(self.data_dict.keys()):
            for example in self.data_dict[key]:
                self.df.loc[count] = {'text': example, 'label': self.text_to_label[key]}
                count = count + 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

def collate_fn(batch):
    sent = []
    label = []
    for idx, data_dict in enumerate(batch):
        sent.append(data_dict['text'])
        label.append(data_dict['label'])
    return sent, label

class ReadabilityGlueDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn_sst2(batch):
    sent = []
    label = []
    for example in batch:
        sent.append(example[0])
        label.append(example[1])
    return sent, label

def collate_fn_stsb(batch):
    sent1 = []
    sent2 = []
    label = []
    for example in batch:
        sent1.append(example[0])
        sent2.append(example[1])
        label.append(example[2])
    return sent1, sent2, label

class ReadabilityTaggingDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn_tagging(batch):
    sent = []
    label = []
    for example in batch:
        sent.append(example[0])
        label.append(example[1])
    return sent, label

def collate_fn_roc(batch):
    sent = []
    label = []
    for example in batch:
        sent.append(example[0])
        label.append(example[1])
    return sent, label
