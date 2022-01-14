import torch
from models import *

def embedding_process(batch, tokenizer, vectors):
    length = [len(tokenizer(x)) for x in batch]
    max_length = max(length)
    data_embedding = torch.zeros(max_length, len(batch), 300)
    for sentence_idx, sentence in enumerate(batch):
        for word_idx, word in enumerate(tokenizer(sentence)):
            data_embedding[word_idx, sentence_idx, :] = vectors[word.text]
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(data_embedding, length, enforce_sorted=False)
    return packed_sequence

def embedding_process_tagging(batch, vectors):
    length = [len(x) for x in batch]
    max_length = max(length)
    data_embedding = torch.zeros(max_length, len(batch), 300)
    for sentence_idx, sentence in enumerate(batch):
        for word_idx, word in enumerate(sentence):
            data_embedding[word_idx, sentence_idx, :] = vectors[word]
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(data_embedding, length, enforce_sorted=False)
    return packed_sequence

def model_chose(task):
    if task in ['sst2']:
        model = LSTMForSingleSequenceClassification(task, num_labels=2)
    if task in ['stsb']:
        model = LSTMForBinarySequenceClassification(task, num_labels=1)
    if task in ['qnli', 'rte', 'mrpc']:
        model = LSTMForBinarySequenceClassification(task, num_labels=2)
    if task in ['roc']:
        model = LSTMForMultipleChoice(num_labels=2)
    return model
