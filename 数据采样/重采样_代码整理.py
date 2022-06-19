#第一步是不是应该搞清楚，共有变量和单独变量，来确定数据的表示方法
#变量命名，变量注释
#代码就和写作业一样，需要整洁、清晰
import re
from collections import Counter

import numpy as np


class resample_func:
    def __init__(self,inpath,outpath):
        self.inpath=inpath
        self.outpath=outpath
    def read_file(self):
        f=open(self.inpath,encoding='utf_8').read()
        flist=re.split('\n\n',f)
        x,y=[[] for i in range(len(flist))],[[] for i in range(len(flist))]
        bio_y=[[]for i in range(len(flist))]
        for n,sent in enumerate(flist):
            for sen in sent.split('\n'):
                x[n].append(sen.split(' ')[0])
                y[n].append(sen.split(' ')[-1][2:])
                bio_y[n].append(sen.split(' ')[-1])

        return x,y,bio_y

    def calweight(self,y):
        lab=Counter([i for j in y for i in j])
        weights={}
        for k in list(lab.keys()):
            weights[k]=-np.log(lab[k]/len([i for j in y for i in j]))
        return weights
    def sample(self,weights,y,method):
        sc_nums,scr_nums,scrd_nums,nscrd_nums=[],[],[],[]
        # resample time at least to be 1
        if method=='sc':
            for sent in y:
                base=1+len([i for i in sent if i!=''])
                sc_nums.append(base)
            final_nums=sc_nums
        elif method=='scr':
            for sent in y:
                l=Counter([i for i in sent if i !=''])#BI label
                base=1+np.ceil(np.log(sum([weights[k]*l[k] for k in l])))
                scr_nums.append(base if base!=-np.inf else 1)
            final_nums=scr_nums
        elif method=='scrd':
            for sent in y:
                l = Counter([i for i in sent if i != ''])
                base=1+np.ceil(sum([weights[k] * l[k] for k in l])/(np.log(len(sent))))
                scrd_nums.append(base if base!=-np.inf else 1)
            final_nums=scrd_nums
        elif method=='nscrd':
            for sent in y:
                l = Counter([i for i in sent if i != ''])
                base = 1 + np.ceil(sum([weights[k] * np.log(l[k]) for k in l]) / (np.log(len(sent))))
                nscrd_nums.append(base if base!=-np.inf else 1)
            final_nums=nscrd_nums

        return final_nums
    def writefile(self,final_nums,x,bio_y):
        f=open(self.outpath,encoding='utf_8',mode='w+')
        final_nums=np.array(final_nums,dtype=np.int)
        for n,num in enumerate(final_nums):
            for i in range(num):
                for j in range(len(x[n])):
                    f.write(('%s\t%s\n')%(x[n][j],bio_y[n][j]))
                f.write('\n')

    def BUS_sample(self,x,y,pos):
        #the select total tokens=pnums
        fmask=[]
        for sent in y:
            pnums=pos*len([i for i in sent if i != ''])
            nnums=len([i for i in sent if i==''])
            if nnums<=pnums:
                mask=[1]*len(sent)
                fmask.append(mask)
            else:
                mask=[1 if i!='' else 0 for i in sent]
                while sum(mask)<pnums:
                    index=np.where(np.array(mask)==1)[0]
                    for i in index:
                        if sum(mask)<pnums and i!=0:
                            mask[i-1]=1
                    for j in index:
                        if sum(mask)<pnums and j!=(len(sent)-1):
                            mask[j+1]=1
                fmask.append(mask)
        return fmask
    def BUS_sample2(self,x,y,pos):
        #the select total neg tokens=pnums
        fmask=[]
        for sent in y:
            pnums=pos*len([i for i in sent if i != ''])
            nnums=len([i for i in sent if i==''])
            if nnums<=pnums:
                mask=[1]*len(sent)
                fmask.append(mask)
            else:
                mask=[1 if i!='' else 0 for i in sent]
                ori=sum(mask)
                while sum(mask)-ori<pnums:
                    index=np.where(np.array(mask)==1)[0]
                    for i in index:
                        if sum(mask)-ori<pnums and i!=0:
                            mask[i-1]=1
                    for j in index:
                        if sum(mask)-ori<pnums and j!=(len(sent)-1):
                            mask[j+1]=1
                fmask.append(mask)
        return fmask

    def buswritefile(self,fmask,x,bio_y):
        f = open(self.outpath, encoding='utf_8', mode='w+')
        for n, num in enumerate(fmask):
            for i in range(len(num)):
                if fmask[n][i]!=0:
                    f.write(('%s\t%s\n') % (x[n][i], bio_y[n][i]))
            f.write('\n')





if __name__=='__main__':
    inpath='../Data_Augmentation/ConLL03/train.txt'
    outpath='../Data_Augmentation/output/ree_bus_train2.txt'
    fun=resample_func(inpath,outpath)
    x,y,bio_y=fun.read_file()
    # weights=fun.calweight(y)
    # sample_nums=fun.sample(weights,y,'scr')
    # print(sample_nums[:5])
    # fun.writefile(sample_nums,x,bio_y)
    # fmask=fun.BUS_sample(x,y,3)
    # fun.buswritefile(fmask, x, bio_y)
    fmask2=fun.BUS_sample2(x,y,3)
    fun.buswritefile(fmask2, x, bio_y)
