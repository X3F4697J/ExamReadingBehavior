import torch
from more_itertools import sort_together
import readability
from tqdm import tqdm
import datasets
from models import *
from data_form import *
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from torchtext.vocab import GloVe, Vectors
from matplotlib import pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizer, BertForSequenceClassification
import numpy as np
from readability_assessment import *
from training_utils import *
from sklearn.metrics import matthews_corrcoef
from roc_story import *

def stsb_value_map(label):
    map_label = []
    for x in label:
        if x == 0 :
            map_label.append(0)
        if (x>0)&(x<=1):
            map_label.append(1)
        if (x>1)&(x<=2):
            map_label.append(2)
        if (x>2)&(x<=3):
            map_label.append(3)
        if (x>3)&(x<=4):
            map_label.append(4)
        if (x>4)&(x<=5):
            map_label.append(5)
    return map_label


vectors = Vectors('')
nlp = English()
tokenizer = Tokenizer(nlp.vocab)
random_seed = 43
task = 'roc'
assessement = 'neural'
type = 'variance'
if assessement == 'uid':
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    gpt2_model.eval()
if assessement == 'neural':
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3).cuda()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model.load_state_dict(torch.load(""))
    bert_model.eval()
epoch = 20
print('******************* Loading Dataset *******************')
train_dataset = ROCDataset("")
test_dataset = ROCDataset("")
metric = datasets.load_metric('glue', 'sst2')
# data:[[[prompt1-4, choice0, choice1], label]]
if assessement == 'flesch':
    hard_train_data, medium_train_data, easy_train_data, whole_train_data = form_data_roc_flesch(train_dataset)
    hard_evaluation_data, medium_evaluation_data, easy_evaluation_data, whole_evaluation_data = form_data_roc_flesch(test_dataset)
if assessement == 'uid':
    easy_train_data, medium_train_data, hard_train_data, whole_train_data = form_data_roc_uid(train_dataset, type, gpt2_tokenizer, gpt2_model)
    easy_evaluation_data, medium_evaluation_data, hard_evaluation_data, whole_evaluation_data = form_data_roc_uid(test_dataset, type, gpt2_tokenizer, gpt2_model)
    gpt2_model = gpt2_model.cpu()
if assessement == 'neural':
    hard_train_data, medium_train_data, easy_train_data, whole_train_data = form_data_roc_neural(train_dataset, bert_model, bert_tokenizer)
    hard_evaluation_data, medium_evaluation_data, easy_evaluation_data, whole_evaluation_data = form_data_roc_neural(test_dataset, bert_model, bert_tokenizer)
    bert_model = bert_model.cpu()
# data:[[[prompt1-4, choice0, choice1], label]]
last_dict = {'easy':0, 'medium':0, 'hard':0}
best_dict = {'easy':0, 'medium':0, 'hard':0}
#evaluation_dict = {'sst2':50, 'rte':5, 'mrpc':5, 'qnli':200,'stsb':15}
evaluation_dict = {'sst2':15, 'rte':5, 'mrpc':5, 'qnli':200, 'stsb':15, 'roc':5}
for hardness_idx, train_data in enumerate([easy_train_data, medium_train_data, hard_train_data]):
    print('******************* Loading Model *******************')
    train_dataset = ReadabilityGlueDataset(train_data)
    test_dataset = ReadabilityGlueDataset(hard_evaluation_data)
    # data:[[[prompt1-4, choice0, choice1], label]]
    hard_train_set, _ = torch.utils.data.random_split(train_dataset, [len(train_dataset), 0], generator=torch.Generator().manual_seed(random_seed))
    eval_set, _ = torch.utils.data.random_split(test_dataset, [len(test_dataset), 0], generator=torch.Generator().manual_seed(random_seed))
    if task in ['stsb', 'mrpc', 'rte', 'wnli', 'qnli', 'mnli', 'qqp']:
        collate_fn = collate_fn_stsb
    elif task in ['sst2', 'cola']:
        collate_fn = collate_fn_sst2
    elif task in ['roc']:
        collate_fn = collate_fn_roc
# collate_fn:batch->[[prompt1-4, choice0, choice1]], [label]
    train_loader = torch.utils.data.DataLoader(hard_train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)  # dataset for training
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=32, shuffle=True, collate_fn=collate_fn)  # dataset for evaluation
    last_result_list = []
    best_result_list = []
    for seed in [7800, 8321, 7084, 8147, 15000]:
        print('******************* Training Random Seed ' + str(seed) + ' *******************')
        torch.manual_seed(seed)
        model = model_chose(task).cuda()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)
        loss_func = torch.nn.MSELoss() if task=='stsb' else torch.nn.CrossEntropyLoss()
        loss_list = []
        accu_list = []
        model.train()
        for _ in tqdm(range(epoch)):
            print('******************* Training Epoch ' + str(_) + ' *******************')
            for batch_idx, train_batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                if task in ['sst2', 'cola']:
                    sent, label = train_batch
                    embedding = embedding_process(sent, tokenizer, vectors)
                    output = model(embedding.cuda())
                    label = torch.tensor(label).cuda()
                if task in ['stsb', 'mrpc', 'rte', 'wnli', 'qnli', 'mnli', 'qqp']:
                    sent1, sent2, label = train_batch
                    embedding1 = embedding_process(sent1, tokenizer, vectors)
                    embedding2 = embedding_process(sent2, tokenizer, vectors)
                    output = model(embedding1.cuda(), embedding2.cuda())
                    if task in ['stsb']:
                        #label = stsb_value_map(label)
                        label = torch.tensor(label).cuda()/5
                    else:
                        label = torch.tensor(label).cuda()
                if task in ['roc']:
                    sent, label = train_batch
                    sent_tmp = []
                    for sentence in sent:
                        prompt = sentence[0] + sentence[1] + sentence[2] + sentence[3]
                        sent_tmp.append(prompt + sentence[4])
                        sent_tmp.append(prompt + sentence[5])
                    embedding = embedding_process(sent_tmp, tokenizer, vectors)
                    output = model(embedding.cuda())
                loss = loss_func(output, torch.tensor(label).cuda())
                loss.backward()
                optimizer.step()
                if (batch_idx % evaluation_dict[task] == 0) & (batch_idx != 0):
                    with torch.no_grad():
                        model.eval()
                        predict_label = []
                        test_label = []
                        for eval_batch in tqdm(eval_loader):
                            if task in ['sst2', 'cola']:
                                sent, label = eval_batch
                                test_label.extend(label)
                                embedding = embedding_process(sent, tokenizer, vectors)
                                label = torch.tensor(label).cuda()
                                output = model(embedding.cuda())
                            if task in ['stsb', 'mrpc', 'rte', 'wnli', 'qnli', 'mnli', 'qqp']:
                                sent1, sent2, label = eval_batch
                                embedding1 = embedding_process(sent1, tokenizer, vectors)
                                embedding2 = embedding_process(sent2, tokenizer, vectors)
                                if task in ['stsb']:
                                    #label = stsb_value_map(label)
                                    #label = torch.tensor(label).cuda()/5
                                    label = torch.tensor(label).cuda()/5
                                else:
                                    label = torch.tensor(label).cuda()
                                test_label.extend(label.cpu().tolist())
                                output = model(embedding1.cuda(), embedding2.cuda())
                            if task in ['roc']:
                                sent, label = eval_batch
                                test_label.extend(label)
                                sent_tmp = []
                                for sentence in sent:
                                    prompt = sentence[0] + sentence[1] + sentence[2] + sentence[3]
                                    sent_tmp.append(prompt + sentence[4])
                                    sent_tmp.append(prompt + sentence[5])
                                embedding = embedding_process(sent_tmp, tokenizer, vectors)
                                label = torch.tensor(label).cuda()
                                output = model(embedding.cuda())
                            if task != 'stsb':
                                predict = output.argmax(dim=1).tolist()
                            else:
                                #predict = output.argmax(dim=1).tolist()
                                predict = output.cpu()[:,0].tolist()
                            predict_label.extend(predict)
                        #pdb.set_trace()
                        results = metric.compute(predictions=predict_label, references=test_label)
                        if task in ['mrpc', 'qqp']:
                            eval_metric = 'f1'
                        if task =='stsb':
                            eval_metric = 'pearson'
                        if task == 'cola':
                            eval_metric = 'matthews_correlation'
                        if task in ['sst2', 'rte', 'qnli','mnli','roc']:
                            eval_metric = 'accuracy'
                        accu_list.append(results[eval_metric])
                    model.train()
        print('last report result of seed '+str(seed)+' :')
        print(accu_list[-1])
        last_result_list.append(accu_list[-1])
        print('best result result of seed '+str(seed)+' :')
        print(max(accu_list))
        best_result_list.append(max(accu_list))
    print('results over 5 run last report :'+str(sum(last_result_list)/len(last_result_list)))
    print('results over 5 run last best :'+str(sum(best_result_list)/len(best_result_list)))
    if hardness_idx == 0:
        last_dict['easy'] = sum(last_result_list)/len(last_result_list)
        best_dict['easy'] = sum(best_result_list)/len(best_result_list)
    if hardness_idx == 1:
        last_dict['medium'] = sum(last_result_list)/len(last_result_list)
        best_dict['medium'] = sum(best_result_list)/len(best_result_list)
    if hardness_idx == 2:
        last_dict['hard'] = sum(last_result_list)/len(last_result_list)
        best_dict['hard'] = sum(best_result_list)/len(best_result_list)

print(last_dict)
print(best_dict)
print(task)
print('lstm')
print(assessement)
if assessement == 'uid':
    print(type)
