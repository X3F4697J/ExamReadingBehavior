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
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertForSequenceClassification, BertTokenizer, BertConfig, get_linear_schedule_with_warmup, BertForMultipleChoice
import numpy as np
from readability_assessment import *
from training_utils import *
from roc_story import *

random_seed = 43
task = 'roc'
epoch = 3
max_length = 256
print('******************* Loading Dataset *******************')
train_dataset = ROCDataset("")
test_dataset = ROCDataset("")
metric = datasets.load_metric('glue', 'sst2')
model_type = 'bert-base-uncased'
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
last_dict = {'easy':0, 'medium':0, 'hard':0}
best_dict = {'easy':0, 'medium':0, 'hard':0}
#evaluation_dict = {'sst2':50, 'rte':5, 'mrpc':5, 'qnli':200, 'stsb':15}
evaluation_dict = {'sst2':15, 'rte':5, 'mrpc':5, 'qnli':200, 'stsb':15, 'roc':5}
for hardness_idx, train_data in enumerate([easy_train_data, medium_train_data, hard_train_data]):
    print('******************* Loading Model *******************')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    configuration = BertConfig.from_pretrained('bert-base-uncased')
    config_dict = configuration.to_dict()
    config_dict['max_seq_length'] = max_length
    if task in ['stsb']:
        config_dict['num_labels'] = 1
    configuration=configuration.from_dict(config_dict)
    if task in ['stsb', 'mrpc', 'rte', 'wnli', 'qnli', 'mnli', 'qqp']:
        collate_fn = collate_fn_stsb
    elif task in ['sst2', 'cola']:
        collate_fn = collate_fn_sst2
    elif task in ['roc']:
        collate_fn = collate_fn_roc
    train_dataset = ReadabilityGlueDataset(train_data)
    test_dataset = ReadabilityGlueDataset(hard_evaluation_data)
    train_set, _ = torch.utils.data.random_split(train_dataset, [len(train_dataset), 0], generator=torch.Generator().manual_seed(random_seed))
    eval_set, _ = torch.utils.data.random_split(test_dataset, [len(test_dataset), 0], generator=torch.Generator().manual_seed(random_seed))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    last_result_list = []
    best_result_list = []
    #for seed in [7800, 8321, 7084, 8147, 15000]:
    for seed in [7800, 8321, 7084, 8147, 15000, 51000, 45678, 11834]:
    #for seed in [7800, 8321]:
        torch.manual_seed(seed)
        print('******************* Training Random Seed ' + str(seed) + ' *******************')
        if task in ['stsb']:
            model = BertForSequenceClassification.from_pretrained(model_type, num_labels=1).cuda()
        elif task in ['roc']:
            model = BertForMultipleChoice.from_pretrained(model_type).cuda()
        else:
            model = BertForSequenceClassification.from_pretrained(model_type).cuda()
        t_total = len(train_loader) * epoch
        warmup_steps = int(t_total * 0.1)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        loss_list = []
        accu_list = []
        model.train()
        for _ in tqdm(range(epoch)):
            print('******************* Training Epoch '+str(_)+' *******************')
            for batch_idx, train_batch in enumerate(tqdm(train_loader)):
                if task in ['sst2', 'cola']:
                    sent, label = train_batch
                    input = tokenizer(sent, padding=True, max_length=max_length, truncation=True, return_tensors='pt')
                if task in ['stsb', 'mrpc', 'rte', 'wnli', 'qnli', 'mnli', 'qqp']:
                    sent1, sent2, label = train_batch
                    input = tokenizer(sent1,sent2,padding=True, max_length=max_length, truncation=True, return_tensors='pt')
                if task in ['roc']:
                    sent, label = train_batch
                    sent_tmp = []
                    pair_tmp = []
                    for sentence in sent:
                        prompt = sentence[0] + sentence[1] + sentence[2] + sentence[3]
                        sent_tmp.append(prompt)
                        sent_tmp.append(prompt)
                        pair_tmp.append(sentence[4])
                        pair_tmp.append(sentence[5])
                    input = tokenizer(sent_tmp, pair_tmp, padding=True, max_length=max_length, truncation=True, return_tensors='pt')
                    input = {key: val.view(int(val.shape[0]/2), 2, val.shape[1]) for key, val in input.items()}
                label = torch.tensor(label)
                output = model(input_ids=input['input_ids'].cuda(), attention_mask=input['attention_mask'].cuda(), token_type_ids=input['token_type_ids'].cuda(), labels=label.cuda())
                loss = output.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if (batch_idx % evaluation_dict[task] == 0) & (batch_idx != 0):
                    model.eval()
                    predict_label = []
                    test_label = []
                    print('******************* Evaluation *******************')
                    with torch.no_grad():
                        for eval_batch in tqdm(eval_loader):
                            if task in ['sst2', 'cola']:
                                sent, label = eval_batch
                                input = tokenizer(sent, padding=True, max_length=max_length, return_tensors='pt')
                            if task in ['stsb', 'mrpc', 'rte', 'wnli', 'qnli', 'mnli', 'qqp']:
                                sent1, sent2, label = eval_batch
                                input = tokenizer(sent1,sent2,padding=True, max_length=max_length, return_tensors='pt')
                            if task in ['roc']:
                                sent, label = eval_batch
                                sent_tmp = []
                                pair_tmp = []
                                for sentence in sent:
                                    prompt = sentence[0] + sentence[1] + sentence[2] + sentence[3]
                                    sent_tmp.append(prompt)
                                    sent_tmp.append(prompt)
                                    pair_tmp.append(sentence[4])
                                    pair_tmp.append(sentence[5])
                                input = tokenizer(sent_tmp, pair_tmp, padding=True, max_length=max_length, truncation=True, return_tensors='pt')
                                input = {key: val.view(int(val.shape[0]/2), 2, val.shape[1]) for key, val in input.items()}
                            test_label.extend(label)
                            label = torch.tensor(label)
                            output = model(input_ids=input['input_ids'].cuda(), attention_mask=input['attention_mask'].cuda(), token_type_ids=input['token_type_ids'].cuda(), labels=label.cuda())
                            if task != 'stsb':
                                predict = output.logits.argmax(dim=1).tolist()
                            else:
                                predict = output.logits.cpu()[:, 0].tolist()
                            predict_label.extend(predict)
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
print('bert')
print(assessement)
if assessement == 'uid':
    print(type)
