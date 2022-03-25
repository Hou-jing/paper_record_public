#tokens_id,atten_mask,----bert

label_map={'O':0}
idx=0
with open('example.train',encoding='utf_8') as f:
    for item in f.readlines()[:100]:
        label=item.strip().split()
        if(len(label)>0):
            label=label[-1]
        else:
            continue
        if(label not in label_map):
            idx+=1
            label_map[label]=idx
print(label_map)
def load_data_from_source(file_name):
    list_data = []
    with open(file_name, 'r',encoding='utf_8') as f:
        list_token = []
        list_label = []
        for item in f.readlines()[:100]:
            if (item.strip() == ""):
                list_data.append({'tokens': list_token, "labels": list_label})
                list_token = []
                list_label = []
            else:
                arr = item.strip().split()
                if (len(arr) == 1):
                    continue
                list_token.append(arr[0])
                list_label.append(arr[1])
    return list_data


train_ds = load_data_from_source(file_name="example.train")
print(train_ds[0])
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
model_name='bert-base-chinese'
tokenizer=BertTokenizer.from_pretrained(model_name)
max_length=128
import torch
import logging
import os
import copy
import json
from torch.utils.data import DataLoader,Dataset,TensorDataset

def tokenize_and_align_labels(examples, tokenizer, label_map):
    tokens_batch, segments_batch, att_batch, labels_,length,tokens_raw = [], [], [], [],[],[]
    for example in examples:
        tokens=example['tokens']
        labels=example['labels']
        labels=[label_map[item] for item in labels]
        tokenized_input = tokenizer(
            tokens,
            return_length=True,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt',
            is_split_into_words=True
            )
        tokenized_input['raw_data']=example['tokens']
        tokens_raw.append(tokenized_input)
        # -2 for [CLS] and [SEP]
        # tokenized_input['labels']=example['labels']
        input_len = len(labels)
        # Zero-pad up to the sequence length0
        input_ids=tokenized_input['input_ids']
        input_mask=tokenized_input['attention_mask']
        segment_ids=tokenized_input['token_type_ids']
        pad_len = max_length - input_ids.shape[1]
        pad_seq = torch.zeros(1, pad_len)
        input_ids = torch.cat((input_ids, pad_seq), dim=-1)
        segment_ids = torch.cat((segment_ids, pad_seq), dim=-1)
        input_mask=torch.cat((input_mask, pad_seq), dim=-1)
        labels=[label_map['O']] + labels + [label_map['O']]
        label_ids=torch.tensor(labels)
        label_ids=label_ids.reshape(1,-1)
        label_ids=torch.cat((label_ids, pad_seq), dim=-1)
        length.append('length')
        tokens_batch.append(input_ids)
        segments_batch.append(segment_ids)
        att_batch.append(input_mask)
        labels_.append(label_ids)
    return tokens_batch, segments_batch, att_batch, labels_,length,tokens_raw

tokens_batch, segments_batch, att_batch, labels,length,tokens_raw=tokenize_and_align_labels(train_ds,tokenizer,label_map)

tokens_batch=torch.cat([l for l in tokens_batch]).int()
print(tokens_batch.shape)
segments_batch=torch.cat([l for l in segments_batch]).int()#sents_length*128
att_batch=torch.cat([l for l in att_batch]).int()
labels=torch.cat([l for l in labels]).int()
dataset=TensorDataset(tokens_batch, segments_batch, att_batch, labels)
train_loader=DataLoader(dataset,shuffle=False,batch_size=2)
print(att_batch.shape)
import json

import torch
from torch import nn
from tqdm import tqdm
from transformers import BertModel,BertTokenizer
from torch.utils.data import TensorDataset,DataLoader
from torch.nn import functional as F

# type2id=json.load(open('type2id.json'))
model_name='bert-base-cased'
#序列标注判断
from torchcrf import CRF
embedding_dim=768
hidden_dim=128
batch_size=2
class E2EModel(nn.Module):
    def __init__(self):
        super(E2EModel, self).__init__()
        self.tagset_size=len(label_map.keys())
        self.hidden_dim=hidden_dim
        self.encode=BertModel.from_pretrained(model_name)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
        #                     num_layers=1, bidirectional=True)
        self.classifier = nn.Linear(768, 128)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf=CRF(num_tags=len(label_map.keys()),batch_first=True)

    def forward(self,inputs_id,att_mask,labels):
        x=self.encode(inputs_id,att_mask)[0]#B*L*768
        # x=x.view(128, -1, 768)
        x=self.classifier(x)
        # x=x.view(128, hidden_dim)
        x=self.hidden2tag(x)
        logits=x
        output = (x,)

        loss = self.crf(emissions=logits, tags=labels.long(), mask=att_mask.byte())
        outputs = (-1 * loss,) + output
        return outputs  # (loss), scores
        # x=x.reshape(128,-1,len(label_map.keys()))
        # att_mask=att_mask.reshape(128,batch_size)
        # labels=labels.reshape(128,-1).long()
        # # x=x.reshape(-1,0)
        # out=self.crf(x,labels,att_mask)
        #
        # return x,out
model=E2EModel()
device='cuda' if torch.cuda.is_available() else 'cpu'
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
for epoch in range(1):
    model=model.to(device=device)
    pres=[]
    for i,data in enumerate(tqdm(train_loader)):

        tokens_batch, segments_batch, att_batch, labels = data
        tokens_batch, segments_batch, att_batch, labels=tokens_batch.to(device), segments_batch.to(device),\
                                                        att_batch.to(device), labels.to(device)
        # 清除梯度
        optimizer.zero_grad()
        outputs=model(tokens_batch, att_batch, labels)
        tmp_eval_loss, logits = outputs[:2]
        # 准备一个batch(batch_size=1)的训练数据
        pre=model.crf.decode(logits,att_batch.byte())
        pres.extend(pre)
        # 计算梯度，更新参数
        tmp_eval_loss.backward()
        optimizer.step()
        print(tmp_eval_loss.item())
        print(pres)
        print(labels.cpu())









