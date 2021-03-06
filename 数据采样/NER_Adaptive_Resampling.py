# -*- coding: utf-8 -*-


from collections import Counter
import re
from math import log,sqrt, ceil#Math.ceil () 函数返回大于或等于一个给定数字的最小整数
import emoji
from tkinter import _flatten


class NER_Adaptive_Resampling():
    
    def __init__(self, inputpath, outputpath):
        self.inputpath = inputpath
        self.outputpath = outputpath
        
    
    def conll_data_read(self):
        
        # Load data in CoNLL format
        f = re.split('\n\t\n|\n\n|\n \n',open(self.inputpath,'r',encoding = 'utf-8').read())[:-1]
        x,y = [[] for i in range(len(f))],[[] for i in range(len(f))]
        for sen in range(len(f)):
            w = f[sen].split('\n')
            for line in w:
                        # Additional data cleaning: transform emoji into text, noisy text oridented.
                        x[sen].append(emoji.demojize(line.split(' ')[0]))#demojize解锁表情
                        y[sen].append(line.split(' ')[-1])
        return x,y#x存储的是token，y存储的是label
    '''x=[['-DOCSTART-'], ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], ['Peter', 'Blackburn'], 
    y=[['O'], ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O'], ['B-PER', 'I-PER']
    '''
    def get_stats(self):
        
        # Get stats of the class distribution of the dataset
        labels = list(_flatten(self.conll_data_read()[-1]))
        num_tokens = len(labels)
        ent = [label[2:] for label in labels if label != 'O']
        count_ent = Counter(ent)#计数，返回字典形式{key:value}
        for key in count_ent:
            #Use frequency instead of count
            count_ent[key] = count_ent[key]/num_tokens
        return count_ent
    
    def resamp(self, method):
        
        # Select method by setting hyperparameters listed below:
        # sc: the smoothed resampling incorporating count
        # sCR: the smoothed resampling incorporating Count & Rareness
        # sCRD: the smoothed resampling incorporating Count, Rareness, and Density
        # nsCRD: the normalized and smoothed  resampling  incorporating Count, Rareness, and Density
        
        if method not in ['sc','sCR','sCRD','nsCRD']:
            raise ValueError("Unidentified Resampling Method")

        output = open(self.outputpath,'w',encoding = 'utf-8')
        x,y =  self.conll_data_read()
        stats = self.get_stats()
        #stats返回的是每个类型的实体tokens占总tokens的数量
        
        
        for sen in range(len(x)):
            
            # Resampling time can at least be 1, which means sentence without 
            # entity will be reserved in the dataset  
            rsp_time = 1
            sen_len = len(y[sen])
            ents = Counter([label[2:] for label in y[sen] if label != 'O'])
                 # Pass if there's no entity in a sentence
            if ents:
                for ent in ents.keys():
                    # Resampling method selection and resampling time calculation, 
                    # see section 'Resampling Functions' in our paper for details.
                    if method == 'sc':
                        rsp_time += ents[ent]
                    if method == 'sCR' or method == 'sCRD':
                        weight = -log(stats[ent],2)
                        rsp_time += ents[ent]*weight
                    if method == 'nsCRD':
                        weight = -log(stats[ent],2)
                        rsp_time += sqrt(ents[ent])*weight
                if method == 'sCR':
                    rsp_time = sqrt(rsp_time)
                if method == 'sCRD' or method == 'nsCRD':
                    rsp_time = rsp_time/sqrt(sen_len)
                # Ceiling to ensure the integrity of resamling time
                rsp_time = ceil(rsp_time) 
            for t in range(rsp_time):
                for token in range(sen_len):
                    output.write(x[sen][token]+' '+y[sen][token]+'\n')
                output.write('\n')
        output.close()
                            
    def BUS(self):
        # Implementation of Balanced UnderSampling (BUS) mentioned in paper 
        # Balanced undersampling: a novel sentence-based undersampling method 
        # to improve recognition of named entities in chemical and biomedical text
        # Appl Intell (2018) Akkasi et al .
        
        # R parameter is set to 3, as what metioned in this paper.
        output = open(self.outputpath,'w',encoding = 'utf-8')
        x,y =  self.conll_data_read()
        for sen in range(len(x)):
            pos = len([label for label in y[sen] if label != 'O'])
            thres = 3*pos
            mask = [1 if label != 'O' else 0 for label in y[sen] ]
            while pos<thres and pos < len(y[sen]):
                for i in range(len(y[sen])-1):
                    if mask[i] == 1 and mask[i+1] == 0:
                        mask[i+1] = 1
                        pos += 1
                for i in range(len(y[sen])-1,0,-1):
                    if mask[i] == 1 and mask[i-1] == 0:
                        mask[i-1] = 1
                        pos += 1
            for i in range(len(y[sen])):
                if mask[i] == 1:
                    output.write(x[sen][i]+' '+y[sen][i]+'\n')
            output.write('\n')

inputpath='./Data_Augmentation/ConLL03/train.txt'
outputpath='./Data_Augmentation/output/re_train.txt'
bus_outputpath='./Data_Augmentation/output/re_bus_train.txt'
# NER_Adaptive_Resampling(inputpath, outputpath).resamp(method='sc')
NER_Adaptive_Resampling(inputpath, bus_outputpath).BUS()