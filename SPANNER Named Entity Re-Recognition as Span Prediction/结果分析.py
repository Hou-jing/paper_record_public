#实体长度
#实体长度为1，实体长度为2.实体长度为3，实体长度大于3
import json
import pickle
#[label2idx, all_predicts, all_span_words
# fwrite_prob = open('test_prob_test .pkl',mode='rb')
# f=pickle.load(fwrite_prob)
# for l in f[2:]:
#     print(l)
# lines=fwrite_prob.readlines()
# for l in lines:
#     pickle.load(fwrite_prob)
#预测的NER的分析
class NERA:
    def __init__(self,fpath):
        self.fpath=fpath
    def elength(self):
        sentitys=[]
        ldict={0:0,1:0,2:0,3:0}#不同句子长度下预测的正确的实体数
        etdict={}#预测正确的不同长度的实体
        epdict={}#不同类型的实体正确数
        tolpre={0:0,1:0,2:0,3:0}#不同句子长度下预测的总的实体数
        toepre={}#预测不同长度的实体数
        toppre={}#不同类型的实体预测数
        #fpath 是预测文件的输出
        with open(self.fpath,'r',encoding='utf_8') as fp:
            step=0
            for line in fp:
                # if '\t' in line:
                    line=line.strip().split('\t',1)
                    # if line!=['-DOCSTART-']:
                    step+=1
                    ws=line[0].split()
                    if len(line)>1:
                            en=''.join(line[1:])
                            ens=en.split('\t')
                            # sentitys.append(ens)
                            for i in range(len(ens)):
                                e,l,tlabel,plabel=ens[i].split(':: ')
                                el=int(l.split(',')[1])-int(l.split(',')[0])
                                if el not in toepre.keys():
                                    toepre[el] = 1
                                else:
                                    toepre[el] = toepre[el] + 1
                                if tlabel == plabel:
                                    if el not  in etdict.keys():
                                        etdict[el]=1
                                    else:
                                        etdict[el]=etdict[el]+1
                                if tlabel == plabel:
                                    if tlabel not in epdict.keys():
                                        epdict[tlabel]=1
                                    else:
                                        epdict[tlabel]=epdict[tlabel]+1
                                if tlabel == plabel:

                                    if len(ws)<30:
                                        ldict[0]=ldict[0]+1
                                    elif 50>len(ws)>=30:
                                        ldict[1]=ldict[1]+1
                                    elif 100>len(ws)>=50:
                                        ldict[2]=ldict[2]+1
                                    else:
                                        ldict[3]=ldict[3]+1
                                        sentitys.append(ws)
                                if tlabel not in toppre.keys():
                                    toppre[tlabel] = 1
                                else:
                                    toppre[tlabel] = toppre[tlabel] + 1
                                if len(ws) < 30:
                                    tolpre[0] = tolpre[0] + 1
                                elif 50 > len(ws) >= 30:
                                    tolpre[1] = tolpre[1] + 1
                                elif 100 > len(ws) >= 50:
                                    tolpre[2] = tolpre[2] + 1
                                else:
                                    tolpre[3] = tolpre[3] + 1
            print('step',step)
            return sentitys,ldict,etdict,epdict,tolpre,toepre,toppre
#golden NER分析

class GNERA:
    def __init__(self,fpath):
        self.fpath=fpath
    def elength(self):
        sentitys=[]
        tolpre={0:0,1:0,2:0,3:0}#不同句子长度下预测的正确的实体数
        toepre={}#预测正确的不同长度的实体
        toppre={}#不同类型的实体正确数
        #fpath 是预测文件的输出
        with open(self.fpath,'r',encoding='utf_8') as fp:
            fp=json.load(fp)
            print(len(fp))
            for line in fp:
                ws = line["context"].split()

                span_idxLab = line["span_posLabel"]

                for seidx, tlabel in span_idxLab.items():
                    sidx, eidx = seidx.split(';')
                    el = int(eidx)-int(sidx)
                    if el<=4:
                        if el not in toepre.keys():
                            toepre[el] = 1
                        else:
                            toepre[el] = toepre[el] + 1
                    else:
                        if 4 not in toepre.keys():
                            toepre[4]=1
                        else:
                            toepre[4]=toepre[4]+1
                    if tlabel not in toppre.keys():
                        toppre[tlabel] = 1
                    else:
                        toppre[tlabel] = toppre[tlabel] + 1
                    if len(ws) < 30:
                        tolpre[0] = tolpre[0] + 1
                    elif 50 > len(ws) >= 30:
                        tolpre[1] = tolpre[1] + 1
                    elif 100 > len(ws) >= 50:
                        tolpre[2] = tolpre[2] + 1
                    else:
                        sentitys.append(ws)
                        tolpre[3] = tolpre[3] + 1

        return sentitys,tolpre,toepre,toppre



sentitys,ldict,etdict,epdict,tolpre,toepre,toppre=NERA('test_test.txt').elength()
print(len(sentitys))
print('预测情况，不同句子长度，不同实体长度，不同实体类型', ldict,etdict,epdict)
print('真实情况下，不同句子长度，不同实体长度，不同实体类型',tolpre,toepre,toppre )
gsentitys,gtolpre,gtoepre,gtoppre=GNERA('spanner.test').elength()
print('在原始数据中，不同句子长度，不同实体长度，不同实体类型',gtolpre,gtoepre,gtoppre)
print(len(sentitys))
def cal(correct_pred,total_pred,total_golden):
    precision =correct_pred / (total_pred+1e-10)
    recall = correct_pred / (total_golden + 1e-10)
    f1 = precision * recall * 2 / (precision + recall + 1e-10)
    return precision,recall,f1

#不同实体长度下的三个指标值为：
for i in range(1,len(etdict)+1):
    print(etdict[i],toepre[i],gtoepre[i])
    precision,recall,f1=cal(etdict[i],toepre[i],gtoepre[i])
    print(f'在实体长度为{i}的实体下，precision={precision},recall={recall},f1={f1}')

for i in range(len(ldict)):
    precision, recall, f1 = cal(ldict[i], tolpre[i], gtolpre[i])
    print(f'在句子长度为{i}的实体下，precision={precision},recall={recall},f1={f1}')

for i in epdict.keys():
    precision, recall, f1 = cal(epdict[i], toppre[i], gtoppre[i])
    print(f'在类型为{i}的实体下，precision={precision},recall={recall},f1={f1}')