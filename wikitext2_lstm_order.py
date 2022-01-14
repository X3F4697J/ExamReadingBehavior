import datasets
import numpy as np
from tqdm import tqdm
import torch
import mxnet as mx
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizer, BertForSequenceClassification
from readability_assessment import *
from training_utils import *
from data_form import *
import argparse
import pickle

parser = argparse.ArgumentParser(description='Index Train')
parser.add_argument('--criteria', type=str, default='flesch')
parser.add_argument('--type',type=str, default='super-linear')
args = parser.parse_args()

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, trainset, validset, testset):
        self.dictionary = Dictionary()
        self.train = self.tokenize(trainset)
        self.valid = self.tokenize(validset)
        self.test = self.tokenize(testset)

    def tokenize(self, data):
        """Tokenizes a text file."""
        # Add words to the dictionary
        tokens = 0
        for line, readability_score in data:
            words = line.split() + ['<eos>']
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word.lower())

        # Tokenize file content
        ids = np.zeros((tokens,), dtype='int32')
        token = 0
        for line, readability_score in data:
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = self.dictionary.word2idx[word.lower()]
                token += 1
        return mx.nd.array(ids, dtype='int32')

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def get_batch(source, i):
    seq_len = min(5, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

class LSTMForWikitext2(torch.nn.Module):
    def __init__(self, vocab_size):
        super(LSTMForWikitext2, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=300)
        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=100, num_layers=2)
        self.dense = torch.nn.Linear(100, vocab_size)
        #self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, input):
        embedding = self.embedding(input)
        packed_output, (hidden_states, cell_states) = self.lstm(embedding)
        dense_output = self.dense(packed_output)
        #output = self.softmax(dense_output.view(-1, ntokens))
        return dense_output.view(-1, ntokens)
task = 'wikitext-2-v1'
glue_dataset = datasets.load_dataset('wikitext', task)
if args.criteria == 'uid':
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    gpt2_model.eval()
if args.criteria == 'neural':
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3).cuda()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model.load_state_dict(torch.load('saved_modelsonestop_bert.dat'))
    bert_model.eval()
if args.criteria == 'flesch':
    hard_train_data, medium_train_data, easy_train_data, whole_train_data = form_data_flesch(glue_dataset, 'train', task)
    hard_evaluation_data, medium_evaluation_data, easy_evaluation_data, whole_test_data = form_data_flesch(glue_dataset, 'test', task)
if args.criteria == 'uid':
    easy_train_data, medium_train_data, hard_train_data, whole_train_data = form_data_uid(glue_dataset, 'train', args.type, task, gpt2_tokenizer, gpt2_model)
    easy_evaluation_data, medium_evaluation_data, hard_evaluation_data, whole_test_data = form_data_uid(glue_dataset, 'test', args.type, task, gpt2_tokenizer, gpt2_model)
    gpt2_model.cpu()
if args.criteria == 'neural':
    hard_train_data, medium_train_data, easy_train_data, whole_train_data = form_data_neural(glue_dataset, 'train', task, bert_model, bert_tokenizer)
    hard_evaluation_data, medium_evaluation_data, easy_evaluation_data, whole_test_data = form_data_neural( glue_dataset, 'test', task, bert_model, bert_tokenizer)
    bert_model = bert_model.cpu()

best_dict = {'easy':0, 'medium':0, 'hard':0}
epoch = 4
# combination of differnt training order
type1 = [easy_train_data, medium_train_data, hard_train_data]# easy to medium to hard
type2 = [easy_train_data, hard_train_data, medium_train_data]# easy to hard to medium
type3 = [medium_train_data, easy_train_data, hard_train_data]# medium to easy to hard
type4 = [medium_train_data, hard_train_data, easy_train_data]# medium to hard to easy
type5 = [hard_train_data, medium_train_data, easy_train_data]# hard to medium to easy
type6 = [hard_train_data, easy_train_data, medium_train_data]# hard to easy to medium
type7 = [whole_train_data]# the random data
type1_list = [[],[]]# list to save the loss and accuracy for different training order
type2_list = [[],[]]
type3_list = [[],[]]
type4_list = [[],[]]
type5_list = [[],[]]
type6_list = [[],[]]
type7_list = [[],[]]

for data_type, lists in zip([type1, type5, type7],[type1_list,type5_list,type7_list]):
    loss_list_allseed = [[],[],[],[],[]]
    result_list_allseed = [[],[],[],[],[]]
    for seed_idx, seed in enumerate([7800, 8321, 7084, 8147, 15000]):
        print('******************* Training Random Seed ' + str(seed) + ' *******************')
        torch.manual_seed(seed)
        corpus = Corpus(whole_train_data, whole_test_data, whole_test_data)# get the dictionary of the whole train data first
        ntokens = len(corpus.dictionary)
        model = LSTMForWikitext2(ntokens).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        loss_func = torch.nn.CrossEntropyLoss()
        loss_list = loss_list_allseed[seed_idx]
        best_result_list = result_list_allseed[seed_idx]
        context = mx.cpu(0)
        model.train()
        for hardness_idx, train_data in enumerate(data_type):
            corpus = Corpus(train_data, whole_test_data, whole_test_data)
            train_data = batchify(corpus.train, 32).as_in_context(context)
            val_data = batchify(corpus.valid, 32).as_in_context(context)
            test_data = batchify(corpus.test, 32).as_in_context(context)
            for _ in range(epoch):
                print('********** Training Epoch '+str(_)+' **********')
                for batch_idx, i in tqdm(enumerate(range(0, train_data.shape[0] - 1, 5))):
                    optimizer.zero_grad()
                    data, target = get_batch(train_data, i)
                    data = torch.tensor(data.as_np_ndarray().tolist())
                    target = torch.tensor(target.as_np_ndarray().tolist())
                    output = model(data.cuda())
                    loss = loss_func(output, target.clone().detach().cuda())
                    loss.backward()
                    optimizer.step()
                    if (batch_idx % 50 == 0) & (batch_idx != 0):
                        with torch.no_grad():
                            model.eval()
                            print('*********** Evaluation ***********')
                            total_L = 0.0
                            ntotal = 0
                            val_loss_list = []
                            for x in tqdm(range(0, test_data.shape[0] - 1, 5)):
                                data, target = get_batch(test_data, x)
                                data = torch.tensor(data.as_np_ndarray().tolist())
                                target = torch.tensor(target.as_np_ndarray().tolist())
                                output = model(data.cuda())
                                val_loss = loss_func(output, target.clone().detach().cuda())
                                val_loss_list.append(val_loss.item())
                            final_val_loss = sum(val_loss_list)/len(val_loss_list)
                            print(final_val_loss)
                            loss_list.append(final_val_loss)
                        model.train()
            print('best ppl:'+str(np.exp(min(loss_list))))
            best_result_list.append(min(loss_list))
        print('average best ppl over 5 seeds:'+str(np.exp(sum(best_result_list)/len(best_result_list))))
        if hardness_idx == 0:
            best_dict['easy'] = sum(best_result_list)/len(best_result_list)
        if hardness_idx == 1:
            best_dict['medium'] = sum(best_result_list)/len(best_result_list)
        if hardness_idx == 2:
            best_dict['hard'] = sum(best_result_list)/len(best_result_list)
    lists[0] = np.array(loss_list_allseed)
    lists[1] = np.array(result_list_allseed)
print(best_dict)
print(task)
print('lstm')
print(args.criteria)
if args.criteria == 'uid':
    f = open('lstm_accu_loss_list_'+task+'_'+args.criteria+'_'+args.type+'.pkl', 'wb')# dump the result when the criteria is uid
else:
    f = open('lstm_accu_loss_list_'+task+'_'+args.criteria+'.pkl', 'wb')# dump the result when the criteria is not uid
# if you want to dump the loss and accuracy for random training, you should save 'type7' instead of type1-6
pickle.dump([type1_list,type5_list,type7_list], f)
f.close()
