import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
from tkinter.filedialog import askopenfilename, asksaveasfilename
import re
import random



class linear():
    def __init__(self, data_cal, data_sample, target, crit, lloq_crit,**kwargs):
        np.seterr(divide='ignore')
        self.data_cal = data_cal
        self.data_sample = data_sample
        self.x = self.data_cal['x']
        self.y = self.data_cal['y']
        self.nobs = len(self.x)
        self.nlevel = len(self.x.unique())
        self.target = target
        self.crit = crit
        self.lloq_crit = lloq_crit
        self.weight_type = {'1': lambda x,y: np.repeat(1,len(x)),'1/x^0.5': lambda x,y: 1/np.sqrt(x),'1/x': lambda x,y: 1/x,'1/x^2': lambda x,y: 1/x**2,
                      '1/y^0.5': lambda x,y: 1/np.sqrt(y),'1/y': lambda x,y: 1/y,'1/y^2': lambda x,y: 1/y**2}
        for k,v in kwargs.items():
            setattr(self,k,v)
            
    def initialize_selection(self):
        self.selection = False
        self.step_selection = 0
    def terminate_selection(self):
        if self.step_selection > self.nlevel:
            return True
        else: 
            return False
    def reset(self):
        self.lloq += 1
        self.hloq += -1
        self.step_level = -1
        self.nlevel += -1   ####sequencial step

    def beta(self):
        beta = np.zeros([self.nobs,self.nobs])
        df = self.data_cal
        for i in range(self.nobs):
            beta[:,i] = (df.iloc[:,1]-df.iloc[i,1])/(df.iloc[:,0]-df.iloc[i,0])
        return beta
    def tstat(self):
        tstat = self.beta()
        for i in range(self.nobs):
            v = tstat[i,:]
            for ind,beta in enumerate(v):
                if np.isinf(beta) or np.isnan(beta):
                    v[ind] = -np.inf
            u = v[v!=-np.inf]
            tstat[i,:] = ss.t.cdf(v,df = len(u)-1,loc = u.mean(),scale = u.std())
        return tstat
    def tscore(self):
        if self.repeat_weight:
            return np.array([abs(sum(self.tstat()[:,k]/self.x)/sum(1/self.x)-0.5) for k in range(self.nobs)])
        else:
            return np.array([abs(sum(self.tstat()[:,k])-0.5) for k in range(self.nobs)])
    def select_repeat(self):
        score = self.tscore()
        self.select_initial = np.zeros(self.nobs)
        for i in self.x:
            self.select_initial[score == min(score[self.x.values == i])] = 1
    def check_bias(self,score):
        return  max(abs(score[1:])) < self.crit and abs(score[0]) < self.lloq_crit
    def transfer_selection(self,model):
        self.select_final = np.zeros(self.nobs)
        for i in range(self.nobs):
            if self.select_initial[i] == 1:
                if self.y.values[i] not in model.y:
                    self.select_final[i] = 0   ### test
        self.model = model
    def newmodels(self,model_temp,ind,model):
        level = self.x.unique()
        to_del = np.where(level == model_temp.x[ind])[0]
        if ind == 0:
            prev = 0
        else:
            prev = to_del - np.where(level == model_temp.x[ind-1])[0]
        if ind == len(model_temp.x)-1:
            next = 0
        else:
            next = np.where(level == model_temp.x[ind+1])[0] - to_del
        if (prev < 3) and (next < 3):
            x = np.delete(model_temp.x,ind,0)
            y = np.delete(model_temp.y,ind,0)
            w = self.weight_type[self.weight](x,y)
            model_test = model(x,y,w)
            model_test.fit()
            return model_test
        else:
            return None
    def select_level(self,model): ### O(n^2)
        fixlevel = self.fixlevel()
        x = np.array(self.x.loc[self.select_initial == 1],dtype = float)
        y = np.array(self.y.loc[self.select_initial == 1],dtype = float)
        w = self.weight_type[self.weight](x,y)
        model_temp = model(x,y,w)
        model_temp.fit()
        if self.check_bias(model_temp.score(self.target)):
            self.transfer_selection(model_temp)
            self.done = True
        else:
            while True:
                models = [self.newmodels(model_temp,ind,model) for ind in range(model_temp.nobs) if model_temp.x[ind] not in fixlevel]
                models = [m for m in models if m]
                scores = np.array([m.score(self.target).sum()/m.nobs for m in models])
                ind = scores.argmin()
                if models[ind].nobs != model_temp.nobs:
                    model_temp = models[ind]
                    if self.check_bias(model_temp.score(self.target)):
                        self.done = True
                        self.transfer_selection(model_temp)
                        break
                    elif model_temp.nobs < 7:
                        self.transfer_selection(model_temp)
                        break
                else:
                    self.transfer_selection(model_temp)
                    break
    def check_repeat(self): ### O(n^2)
        self.selection = True
        bias = abs(self.target(self.x,self.y,self.model.beta))
        for ind in range(self.nobs):
            value = self.x.values[ind]
            true =  min(bias.loc[self.x.values == value])
            if (self.select_initial[ind] == 1) and (true != bias.values[ind]):
                self.selection = False
                self.select_initial[ind] = 0
            elif (self.select_initial[ind] == 0) and (true == bias.values[ind]):
                self.selection = False
                self.select_initial[ind] = 1
        if not self.selection:
            self.done = False
    def check_nlevel(self):
        fixlevel = self.fixlevel()
        if self.done:
            return False
        elif self.nlevel - self.step_level  < 7:
            if self.selectmode == 'hloq stepdown' or self.selectmode == 'lloq stepup':
                self.reset()
                if self.nlevel < 7:
                    print('nlevel is less than 6, try different weight or selectmode')
                    self.nlevel = len(self.x)
                    return False
                else:
                    print('change lloq or hloq')
                    return True
            else:
                print('nlevel is less than 6, try different weight or selectmode')
                return False
        elif self.rangemode == 'auto':
            df = self.data_sample
            if max(df.loc[df['y'] < max(self.y),'y']) > max(fixlevel) or min(df.loc[df['y'] > min(self.y),'y']) < min(fixlevel):
                print("fixlevel doesn't cover sample range, try different weight or selectmode")
                return False
        else:
            return False
    def circle_selection(self,model):
        self.initialize_selection()
        while not self.selection: ### O(n^3)
            self.select_level(model)
            self.check_repeat()
            self.step_selection += 1
            if self.terminate_selection():
                break
        print(pd.DataFrame({'x':self.model.x,'y':self.model.y,'score': self.model.score(self.target),'weight': self.model.w}))
    def fix(self):
        if len(self.fixpoints) != 0:
            return lambda i: self.fixpoints[i]
        elif self.selectmode == 'hloq stepdown':
            return lambda i: [self.lloq,-(i+1)]
        elif self.selectmode == 'lloq stepup':
            return lambda i: [i,self.hloq]
        elif self.selectmode == 'sequantial stepdownup':
            return lambda i: [int(i/2),-(int((i+1)/2)+1)]
        elif self.selectmode == 'sequantial stepupdown':
            return lambda i: [int((i+1)/2),-(int(i/2)+1)]
    def fixlevel(self):
        return [self.x.unique()[i] for i in self.fix()(self.step_level)]
    def fit(self,model):
        if not self.data_sample:
            print('Sample not supplied, changing rangemode to unlimited')
            self.rangemode = 'unlimited'
        self.select_repeat()
        self.selection = False
        self.done = False
        if self.order == 'weight':
            self.step_level = 0
            while not self.done: ### O(n^4)
                for w in self.weights: 
                    print('weight: {}'.format(w))
                    self.weight = w
                    self.circle_selection(model)
                    if self.done:
                        break
                if self.check_nlevel():
                    break
                else:
                    self.step_level += 1
        elif self.order == 'range':
            for w in self.weights:
                print('weight: {}'.format(w))
                self.weight = w
                self.initialize_selection()
                while not self.done:
                    self.circle_selection(model)
                    if not self.check_nlevel():
                        break
                    else:
                        self.step_level += 1
                if self.done:
                    break
        elif self.order == 'weight only':
            for w in self.weights:
                print('weight: {}'.format(w))
                self.weight = w
                self.circle_selection()
                if self.done:
                    break   




def beta(df,nobs):
    beta = np.zeros([nobs,nobs])
    for j in range(nobs):
        for i in range(nobs):
            if df.iloc[i,0]-df.iloc[j,0] != 0:
                beta[j,i] = (df.iloc[i,1]-df.iloc[j,1])/(df.iloc[i,0]-df.iloc[j,0])
    return beta
def beta_tstat(df,nobs):
    tstat = self.beta(df,nobs)
    def tscore(j,v):
        if j != 0:
            return ss.t.cdf(j,df = len(v)-1,loc = v.mean(),scale = v.std())
        else:
            return 0
    for i in range(self.nobs):
        u = tstat[i,:]
        v = u[u!=0]
        tstat[i,:] = [tscore(j,v) for j in u]
    return tstat
def beta_tscore(df,nobs,repeat_weight):
    if repeat_weight == True:
        return np.array([abs(sum(beta_tstat(df,nobs)[:,k]/df.loc['x'])/sum(1/df.loc['x'])-0.5) for k in range(nobs)])
    else:
        return np.array([abs(sum(self.tstat(df,nobs)[:,k])/sum(1/df.loc['x'])-0.5) for k in range(nobs)])



