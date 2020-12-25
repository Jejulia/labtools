import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
from tkinter.filedialog import askopenfilename, asksaveasfilename
import re
import random


class linear():
    def __init__(self,name,data_cal,data_sample,lloq,hloq,crit,lloq_crit,weights,
                 optimize,target,order,rangemode,selectmode,fixpoints,repeat_weight):
        self.name = name
        self.data_cal = data_cal
        self.data_sample = data_sample
        self.x = self.data_cal['x']
        self.y = self.data_cal['y']
        self.nobs = len(self.x)
        self.nlevel = len(self.x.unique())
        self.step_level = 0
        self.step_selection = 0
        self.selection = False
        self.done = False
        self.lloq = lloq
        self.hloq = hloq
        self.crit = crit
        self.lloq_crit = lloq_crit
        self.weight_type = {'1': lambda x,y: 1,'1/x^0.5': lambda x,y: 1/np.sqrt(x),'1/x': lambda x,y: 1/x,'1/x^2': lambda x,y: 1/x**2,
                      '1/y^0.5': lambda x,y: 1/np.sqrt(y),'1/y': lambda x,y: 1/y,'1/y^2': lambda x,y: 1/y**2}
        self.weights = weights
        self.optimize = optimize
        self.target = target
        self.order = order
        self.rangemode = rangemode
        self.selectmode = selectmode
        self.fixpoints = fixpoints
        self.repeat_weight = repeat_weight
    def initialize_selection(self):
        self.selection = False
        self.step_selection = 0
    def terminate_selection(self):
        if self.step_selection > self.nlevel:
            self.done = True
            return True
        else: 
            return False
    def reset(self):
        self.lloq += 1
        self.hloq += -1
        self.step_level = -1
        self.nlevel += -1
    def vscore(self,x,y,params):
        if self.target == "accuracy":
            return (y-params[0])/params[1]/x
        elif self.target == "residual":
            return (y-params[0])/params[1]-x
    def beta(self):
        beta = np.zeros([self.nobs,self.nobs])
        df = self.data_cal
        for j in range(self.nobs):
            for i in range(self.nobs):
                if df.iloc[i,0]-df.iloc[j,0] != 0:
                    beta[j,i] = (df.iloc[i,1]-df.iloc[j,1])/(df.iloc[i,0]-df.iloc[j,0])
        return beta
    def tstat(self):
        tstat = self.beta().copy() 
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
    def tscore(self):
        if self.repeat_weight == True:
            return np.array([abs(sum(self.tstat()[:,k]/self.x)/sum(1/self.x)-0.5) for k in range(self.nobs)])
        else:
            return np.array([abs(sum(self.tstat()[:,k])-0.5) for k in range(self.nobs)])
    def select_repeat(self):
        score = self.tscore()
        self.select_r = np.zeros(self.nobs)
        for i in self.x:
            self.select_r[score == min(score[self.x.values == i])] = 1
    def check_bias(self,acc):
        if self.target == "accuracy":
            return  max(abs(acc[1:]-1)) < self.crit and abs(acc[0]-1) < self.lloq_crit
        elif self.target == "residual":
            return  max(abs(acc[1:])) < self.crit and abs(acc[0]) < self.lloq_crit
    def transfer_selection(self,model):
        self.select_p = np.zeros(self.nobs)
        k = 0
        while k < len(model.y):
            for j in range(self.nobs):
                if self.y.values[j] == model.y[k]:
                    self.select_p[j] = 1
                    k += 1
                    break
        self.model = model
    def select_level(self,model):
        fixlevel = self.fixlevel()
        x = np.array(self.x.loc[self.select_r == 1],dtype = float)
        y = np.array(self.y.loc[self.select_r == 1],dtype = float)
        w = self.weight_type[self.weight](x,y)
        model_temp = model(x,y,w)
        if self.check_bias(model_temp.vscore(self.target)):
            self.transfer_selection(model_temp)
            self.done = True
        else:
            while True:
                model_win = model_temp
                for i in range(len(model_temp.y)):
                    if model_temp.x[i] not in fixlevel:
                        try:
                            if model_temp.x[i+1,1] < 10*model_temp.x[i-1,1]:
                                x = np.delete(model_temp.x.copy(),i,0)
                                y = np.delete(model_temp.y.copy(),i,0)
                                w = self.weight_type[self.weight](x,y)
                                model_test = model(x,y,w)
                                if model_test.score(self.target) < model_win.score(self.target):
                                    model_win = model_test
                        except:
                            x = np.delete(model_temp.x.copy(),i,0)
                            y = np.delete(model_temp.y.copy(),i,0)
                            w = self.weight_type[self.weight](x,y)
                            model_test = model(x,y,w)
                            if model_test.score(self.target) < model_win.score(self.target):
                                model_win = model_test
                if len(model_win.x) != len(model_temp.x):
                    model_temp = model_win
                    if self.check_bias(model_temp.vscore(self.target)) or len(model_temp.y) < 7:
                        if self.check_bias(model_temp.vscore(self.target)):
                            self.done = True
                        self.transfer_selection(model_temp)
                        break
                else:
                    self.transfer_selection(model_temp)
                    break
    def check_repeat(self):
        self.selection = True
        for k in self.model.x:
            if self.target == "accuracy":
                true =  min(abs(self.vscore(self.x,self.y,self.model.fit.params)-1).loc[self.x.values == k])
                bias = abs(self.vscore(self.x,self.y,self.model.fit.params)-1)
            elif self.target == "residual":
                true =  min(abs(self.vscore(self.x,self.y,self.model.fit.params)).loc[self.x.values == k])
                bias = abs(self.vscore(self.x,self.y,self.model.fit.params))
            for j in range(self.nobs):
                if self.x.values[j] != k:
                    pass
                elif true != bias.values[j]:
                    if self.select_r[j] == 1:
                        self.selection = False
                        self.select_r[j] = 0
                else:
                    self.select_r[j] = 1
            if not self.selection:
                self.done = False
                break
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
    def circle_selection(self,model):
        self.initialize_selection()
        while not self.selection:
            self.select_level(model)
            self.check_repeat()
            self.step_selection += 1
            if self.terminate_selection():
                break
        if self.target == "accuracy":
            print(pd.DataFrame({'x':self.model.x,'y':self.model.y,'accuracy': self.model.vscore(self.target),'weight': self.model.w}))
        elif self.target == "residual":
            print(pd.DataFrame({'x':self.model.x,'y':self.model.y,'residual': self.model.vscore(self.target),'weight': self.model.w}))
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
        if type(self.data_sample) == str:
            print('Sample not supplied, changing rangemode to unlimited')
            self.rangemode = 'unlimited'
        print(self.name)
        self.select_repeat()
        if self.order == 'weight':
            while not self.done:
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
                if self.done == True:
                    break
        elif self.order == 'weight only':
            for w in self.weights:
                print('weight: {}'.format(w))
                self.weight = w
                self.circle_selection()
                if self.done:
                    break   
