import torch
import numpy as np
import readability
from more_itertools import sort_together
from tqdm import tqdm
import pdb
STRIDE = 200


def super_linear_uid_compute(input, k_power):
    uid = torch.zeros(1)
    for surprisal in input:
        uid = uid + torch.pow(surprisal, k_power)
    return uid/len(input)

def variance_uid_compute(input, language_capacity):
    uid = torch.zeros(1)
    for surprisal in input:
        uid = uid + torch.square((surprisal-language_capacity))
    return uid/len(input)

def score_gpt2(sentence, tokenizer, model):
    with torch.no_grad():
        all_log_probs = torch.tensor([], device=model.device)
        offset_mapping = []
        start_ind = 0
        if sentence == '':
            return [], []
        while True:
            encodings = tokenizer(sentence[start_ind:], max_length=1022, truncation=True, return_offsets_mapping=True)
            tensor_input = torch.tensor([[tokenizer.bos_token_id] + encodings['input_ids'] + [tokenizer.eos_token_id]],
                                        device=model.device)
            output = model(tensor_input, labels=tensor_input)
            shift_logits = output['logits'][..., :-1, :].contiguous()
            shift_labels = tensor_input[..., 1:].contiguous()
            log_probs = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                          shift_labels.view(-1), reduction='none')
            assert torch.isclose(torch.exp(sum(log_probs) / len(log_probs)), torch.exp(output['loss']))
            offset = 0 if start_ind == 0 else STRIDE - 1
            all_log_probs = torch.cat([all_log_probs, log_probs[offset:-1]])
            offset_mapping.extend([(i + start_ind, j + start_ind) for i, j in encodings['offset_mapping'][offset:]])
            if encodings['offset_mapping'][-1][1] + start_ind == len(sentence):
                break
            start_ind += encodings['offset_mapping'][-STRIDE][1]
        return np.asarray(all_log_probs.cpu()), offset_mapping

def flesch_data_iter(glue_dataset, dataset_type, task, readability_list ,data_list, sentence_list, label_list):
    if task in ['sst2']:
        for example, label in tqdm(zip(glue_dataset[dataset_type]['sentence'], glue_dataset[dataset_type]['label'])):
            try:
                results = readability.getmeasures(example, lang='en')
                readability_list.append(results['readability grades']['FleschReadingEase'])
                data_list.append([example, label, results['readability grades']['FleschReadingEase']])
                sentence_list.append(example)
                label_list.append(label)
            except ValueError:
                pass
    if task in ['qnli', 'stsb','mrpc','rte']:
        if task in ['stsb','mrpc','rte']:
            sentence1 = 'sentence1'
            sentence2 = 'sentence2'
        if task in ['qnli']:
            sentence1 = 'question'
            sentence2 = 'sentence'
        for example1, example2, label in tqdm(zip(glue_dataset[dataset_type][sentence1], glue_dataset[dataset_type][sentence2], glue_dataset[dataset_type]['label'])):
            try:
                results1 = readability.getmeasures(example1, lang='en')
                results2 = readability.getmeasures(example2, lang='en')
                average_score = (results1['readability grades']['FleschReadingEase'] + results2['readability grades']['FleschReadingEase'])/2
                readability_list.append(average_score)
                data_list.append([example1, example2, label, average_score])
                sentence_list.append([example1, example2])
                label_list.append(label)
            except ValueError:
                pass
    if task in ['wikitext-2-v1']:
        for example in tqdm(glue_dataset[dataset_type]['text']):
            try:
                results = readability.getmeasures(example, lang='en')
                readability_list.append(results['readability grades']['FleschReadingEase'])
                data_list.append([example, results['readability grades']['FleschReadingEase']])
                sentence_list.append(example)
                label_list.append(None)
            except ValueError:
                pass
                # readability_list.append(300)
                # data_list.append([example, 300])
                # sentence_list.append(example)
                # label_list.append(None)
    if task in ['conll2003']:
        for example, label in tqdm(zip(glue_dataset[dataset_type]['tokens'], glue_dataset[dataset_type]['ner_tags'])):
            try:
                sentence = ' '.join(example).replace(' .', '.')
                results = readability.getmeasures(sentence)
                readability_list.append(results['readability grades']['FleschReadingEase'])
                data_list.append([sentence, label, results['readability grades']['FleschReadingEase']])
                sentence_list.append(sentence)
                label_list.append(label)
            except ValueError:
                sentence = ' '.join(example).replace(' .', '.')
                readability_list.append(300)
                data_list.append([sentence, label, results['readability grades']['FleschReadingEase']])
    return readability_list, data_list, sentence_list, label_list

def neural_data_iter(glue_dataset, dataset_type, task, readability_list ,data_list, sentence_list, label_list, model, tokenzier):
    if task in ['sst2']:
        for example, label in tqdm(zip(glue_dataset[dataset_type]['sentence'], glue_dataset[dataset_type]['label'])):
            input = tokenzier(example, return_tensors='pt')
            #pdb.set_trace()
            output = model(input_ids=input['input_ids'].cuda(), token_type_ids=input['token_type_ids'].cuda(), attention_mask=input['attention_mask'].cuda())
            #pdb.set_trace()
            reada_label = output.logits.softmax(dim=1)
            reada_label = reada_label[0][0]+reada_label[0][1]*2+reada_label[0][2]*3
            readability_list.append(reada_label.item())
            data_list.append([example, label, reada_label.item()])
            sentence_list.append(example)
            label_list.append(label)
    if task in ['qnli', 'stsb','mrpc','rte']:
        if task in ['stsb','mrpc','rte']:
            sentence1 = 'sentence1'
            sentence2 = 'sentence2'
        if task in ['qnli']:
            sentence1 = 'question'
            sentence2 = 'sentence'
        for example1, example2, label in tqdm(zip(glue_dataset[dataset_type][sentence1], glue_dataset[dataset_type][sentence2], glue_dataset[dataset_type]['label'])):
            input = tokenzier(example1+example2, max_length=512, padding=True, truncation=True, return_tensors='pt')
            output = model(input_ids=input['input_ids'].cuda(), token_type_ids=input['token_type_ids'].cuda(), attention_mask=input['attention_mask'].cuda())
            reada_label = output.logits.softmax(dim=1)
            reada_label = reada_label[0][0]+reada_label[0][1]*2+reada_label[0][2]*3
            readability_list.append(reada_label.item())
            data_list.append([example1, example2, label, reada_label.item()])
            sentence_list.append([example1, example2])
            label_list.append(label)
    if task in ['wikitext-2-v1']:
        for example in tqdm(glue_dataset[dataset_type]['text']):
            if example != '':
                input = tokenzier(example, max_length=512, padding=True, truncation=True, return_tensors='pt')
                output = model(input_ids=input['input_ids'].cuda(), token_type_ids=input['token_type_ids'].cuda(), attention_mask=input['attention_mask'].cuda())
                reada_label = output.logits.softmax(dim=1)
                reada_label = reada_label[0][0]+reada_label[0][1]*2+reada_label[0][2]*3
                readability_list.append(reada_label.item())
                data_list.append([example, reada_label.item()])
                sentence_list.append([example])
                label_list.append(None)
            else:
                pass
    if task in ['conll2003']:
        for example, label in tqdm(zip(glue_dataset[dataset_type]['tokens'], glue_dataset[dataset_type]['ner_tags'])):
            sentence = ' '.join(example).replace(' .', '.')
            input = tokenzier(sentence, max_length=512, padding=True, truncation=True, return_tensors='pt')
            output = model(input_ids=input['input_ids'].cuda(), token_type_ids=input['token_type_ids'].cuda(), attention_mask=input['attention_mask'].cuda())
            reada_label = output.logits.softmax(dim=1)
            reada_label = reada_label[0][0]+reada_label[0][1]*2+reada_label[0][2]*3
            readability_list.append(reada_label.item())
            data_list.append([sentence, label, reada_label.item()])
            sentence_list.append(sentence)
            label_list.append(label)
    return readability_list, data_list, sentence_list, label_list

def uid_data_iter(glue_dataset, dataset_type, task, data_list, sentence_list, label_list, gpt2model ,tokenizer, uid_score_func, constant):
    score_list = []
    if task in ['sst2']:
        for example, label in tqdm(zip(glue_dataset[dataset_type]['sentence'], glue_dataset[dataset_type]['label'])):
            score, offset_mapping = score_gpt2(example, tokenizer, gpt2model)
            uid_score = uid_score_func(score, constant)
            label_list.append(label)
            sentence_list.append(example)
            score_list.append(uid_score.item())
            data_list.append([example, label, uid_score])
    if task in ['qnli', 'stsb','mrpc','rte']:
        if task in ['stsb','mrpc','rte']:
            sentence1 = 'sentence1'
            sentence2 = 'sentence2'
        if task in ['qnli']:
            sentence1 = 'question'
            sentence2 = 'sentence'
        for example1, example2, label in tqdm(zip(glue_dataset[dataset_type][sentence1], glue_dataset[dataset_type][sentence2], glue_dataset[dataset_type]['label'])):
            score1, offset_mapping = score_gpt2(example1, tokenizer, gpt2model)
            uid_score1 = uid_score_func(torch.tensor(score1), constant)
            score2, offset_mapping = score_gpt2(example2, tokenizer, gpt2model)
            uid_score2 = uid_score_func(torch.tensor(score2), constant)
            label_list.append(label)
            sentence_list.append([example1, example2])
            average_score = uid_score1.item() + uid_score2.item()
            data_list.append([example1, example2,  label, average_score])
            score_list.append(average_score)
            sentence_list.append([example1, example2])
            label_list.append(label)
    if task in ['wikitext-2-v1']:
        for example in tqdm(glue_dataset[dataset_type]['text']):
            if example != '':
                score, offset_mapping = score_gpt2(example, tokenizer, gpt2model)
                uid_score = uid_score_func(score, constant)
                score_list.append(uid_score.item())
                data_list.append([example, uid_score.item()])
                sentence_list.append(example)
                label_list.append(None)
            else:
                pass
    if task in ['conll2003']:
        for example, label in tqdm(zip(glue_dataset[dataset_type]['tokens'], glue_dataset[dataset_type]['ner_tags'])):
            sentence = ' '.join(example).replace(' .', '.')
            score, offset_mapping = score_gpt2(sentence, tokenizer, gpt2model)
            uid_score = uid_score_func(score, constant)
            label_list.append(label)
            sentence_list.append(sentence)
            score_list.append(uid_score.item())
            data_list.append([sentence, label, uid_score])
    return score_list, data_list, sentence_list, label_list

def form_data_uid(glue_dataset, dataset_type, operation_type, task, tokenizer, gpt2model):
    data_list = []
    sentence_list = []
    label_list = []
    if operation_type == 'super-linear':
        uid_score_func = super_linear_uid_compute#scores, 1.25)
        constant = torch.tensor(1.25)
    if operation_type == 'variance':
        uid_score_func = variance_uid_compute#(scores, 3.8845)
        constant = torch.tensor(3.8845)
    score_list, data_list, sentence_list, label_list = uid_data_iter(glue_dataset, dataset_type, task, data_list, sentence_list, label_list, gpt2model,tokenizer, uid_score_func, constant)
    sorted_data = sort_together([score_list, data_list])[1]
    hard, medium, easy = int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data))
    hard_data = sorted_data[0:hard]
    medium_data = sorted_data[hard:hard + medium]
    easy_data = sorted_data[hard + medium:]
    return hard_data, medium_data, easy_data, sorted_data

def form_data_flesch(glue_dataset, dataset_type, task):
    readability_list = []
    data_list = []
    sentence_list = []
    label_list = []
    readability_list, data_list, sentence_list, label_list = flesch_data_iter(glue_dataset, dataset_type, task, readability_list, data_list, sentence_list, label_list)
    try:
        sorted_data = sort_together([readability_list, data_list])[1]
    except IndexError:
        pdb.set_trace()
    hard, medium, easy = int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data))
    hard_data = sorted_data[0:hard]
    medium_data = sorted_data[hard:hard + medium]
    easy_data = sorted_data[hard + medium:]
    return hard_data, medium_data, easy_data, data_list

def form_data_neural(glue_dataset, dataset_type,task, model, tokenizer):
    readability_list = []
    data_list = []
    sentence_list = []
    label_list = []
    readability_list, data_list, sentence_list, label_list = neural_data_iter(glue_dataset, dataset_type, task, readability_list, data_list, sentence_list, label_list, model, tokenizer)
    sorted_data = sort_together([readability_list, data_list])[1]
    hard, medium, easy = int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data))
    hard_data = sorted_data[0:hard]
    medium_data = sorted_data[hard:hard + medium]
    easy_data = sorted_data[hard + medium:]
    return hard_data, medium_data, easy_data, sorted_data

def form_data_roc_flesch(dataset, num_sentence=6):
    readability_list = []
    data_list = []
    sentence_list = []
    label_list = []
    sentence_flattened = []
    label_flattened = []
    for words, label in dataset:
        sentence_flattened.extend(words)
        label_flattened.extend([label for i in range(num_sentence)])
    dataset_fixed = {'train': {'sentence': sentence_flattened, 'label': label_flattened}}
    readability_list, data_list, sentence_list, label_list = flesch_data_iter(dataset_fixed, 'train', 'sst2', readability_list, data_list, sentence_list, label_list)
    data_list = dataset.data
    readability_list_squeezed = []
    for i in range(0, len(readability_list), num_sentence):
        readability_list_squeezed.append(sum(readability_list[i:i+num_sentence])/num_sentence)
    sorted_data = sort_together([readability_list_squeezed, data_list])[1]
    hard, medium, easy = int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data))
    hard_data = sorted_data[0:hard]
    medium_data = sorted_data[hard:hard + medium]
    easy_data = sorted_data[hard + medium:]
    return hard_data, medium_data, easy_data, data_list

def form_data_roc_uid(dataset, operation_type, tokenizer, gpt2model, num_sentence=6):
    data_list = []
    sentence_list = []
    label_list = []
    if operation_type == 'super-linear':
        uid_score_func = super_linear_uid_compute#scores, 1.25)
        constant = torch.tensor(1.25)
    if operation_type == 'variance':
        uid_score_func = variance_uid_compute#(scores, 3.8845)
        constant = torch.tensor(3.8845)
    sentence_flattened = []
    label_flattened = []
    for words, label in dataset:
        sentence_flattened.extend(words)
        label_flattened.extend([label for i in range(num_sentence)])
    dataset_fixed = {'train': {'sentence': sentence_flattened, 'label': label_flattened}}
    score_list, data_list, sentence_list, label_list = uid_data_iter(dataset_fixed, 'train', 'sst2', data_list, sentence_list, label_list, gpt2model,tokenizer, uid_score_func, constant)
    data_list = dataset.data
    score_list_squeezed = []
    for i in range(0, len(score_list), num_sentence):
        score_list_squeezed.append(sum(score_list[i:i + num_sentence]) / num_sentence)
    sorted_data = sort_together([score_list_squeezed, data_list])[1]
    hard, medium, easy = int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data))
    hard_data = sorted_data[0:hard]
    medium_data = sorted_data[hard:hard + medium]
    easy_data = sorted_data[hard + medium:]
    return hard_data, medium_data, easy_data, data_list

def form_data_roc_neural(dataset, model, tokenizer, num_sentence=6):
    readability_list = []
    data_list = []
    sentence_list = []
    label_list = []
    sentence_flattened = []
    label_flattened = []
    for words, label in dataset:
        sentence_flattened.extend(words)
        label_flattened.extend([label for i in range(num_sentence)])
    dataset_fixed = {'train': {'sentence': sentence_flattened, 'label': label_flattened}}
    readability_list, data_list, sentence_list, label_list = neural_data_iter(dataset_fixed, 'train', 'sst2', readability_list, data_list, sentence_list, label_list, model, tokenizer)
    data_list = dataset.data
    readability_list_squeezed = []
    for i in range(0, len(readability_list), num_sentence):
        readability_list_squeezed.append(sum(readability_list[i:i + num_sentence]) / num_sentence)
    sorted_data = sort_together([readability_list_squeezed, data_list])[1]
    hard, medium, easy = int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data)), int(1 / 3 * len(sorted_data))
    hard_data = sorted_data[0:hard]
    medium_data = sorted_data[hard:hard + medium]
    easy_data = sorted_data[hard + medium:]

    return hard_data, medium_data, easy_data, data_list
