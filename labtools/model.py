import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, asksaveasfilename
import re
import multiprocessing as mp
from labtools.plot import plot_calibration_line
import random
import labtools.optimizer as op


def test(a,x):
    if a > x:
        return a
    else:
        pass


def comp(data):
    
    pool = mp.Pool(mp.cpu_count())
    
    results = pool.starmap(test, [(a,10) for a in data])
    
    pool.close()
    return results


class lm():
    def __init__(self,x,y,w):
        self.x = x
        self.y = y
        self.w = w
        self.model = sm.WLS(y,sm.add_constant(x),weights = self.w)
        self.fit = self.model.fit()
    def accuracy(self):
        return (self.y-self.fit.params[0])/self.fit.params[1]/self.x
    def residual(self):
        return (self.y-self.fit.params[0])/self.fit.params[1]-self.x
    def vscore(self,target="accuracy"):
        if target == "accuracy":
            return (self.y-self.fit.params[0])/self.fit.params[1]/self.x
        elif target == "residual":
            return (self.y-self.fit.params[0])/self.fit.params[1]-self.x
    def score(self,target="accuracy"):
        if target == "accuracy":
            return np.dot((self.accuracy()-1),(self.accuracy()-1))/(len(self.y)-1)
        elif target == "residual":
            return np.dot(self.residual(),self.residual())/(len(self.y)-1)


class calibration():
    def __init__(self,df,name,sample = 'none'):
        self.data_cal = df
        self.data_sample = sample
        self.name = name
        self.nobs = len(self.data_cal.index)
        self.nlevel = len(self.data_cal['x'].unique())
    def accuracy(self,x,y,params):
        return (y-params[0])/params[1]/x
    def fit(self,optimize = True,target = "accuracy",crit = 0.15, lloq_crit = 0.2, 
            order = 'range' , rangemode = 'auto', selectmode = 'hloq stepdown',
            lloq = 0, hloq = -1, weights = ['1','1/x^0.5','1/x','1/x^2'], fixpoints = [], model = lm,
            repeat_weight = True):
        # avalible order: 'range', 'weight'
        # available rangemode: 'auto', 'unlimited'
        # available weights: '1','1/x^0.5','1/x','1/x^2','1/y^0.5','1/y','1/y^2'
        # available selectmode: 'hloq stepdown','lloq stepup','sequantial stepdownup','sequantial stepupdown'
        om = op.linear(name = self.name,data_cal = self.data_cal,data_sample = self.data_sample,optimize = optimize,target = target,
                             crit = crit, lloq_crit = lloq_crit, order = order , rangemode = rangemode, selectmode = selectmode,
                             lloq = lloq, hloq = hloq, weights = weights, fixpoints = fixpoints,repeat_weight = repeat_weight)
        om.fit(model)
        self.model = om.model
        self.data_cal['select'] = om.select_p
        self.data_cal['weight'] = om.weight_type[om.weight](om.x,om.y)
        print('done')
        del om
    def quantify(self,rangelimit):
        self.data_sample['x'] = (self.data_sample['y']-self.model.fit.params[0])/self.model.fit.params[1]
        if rangelimit == True:
            for i in self.data_sample['x']:
                if i < min(self.model.x):
                    self.data_sample.loc[self.data_sample['x'] == i,'x'] = '{}(< LLOQ)'.format(round(i,3))
                elif i > max(self.model.x):
                    self.data_sample.loc[self.data_sample['x'] == i,'x'] = '{}(> HLOQ)'.format(round(i,3))
    def drop_level(self,level):
        self.data_cal = self.data_cal.drop(level)
        self.nobs = len(self.data_cal.index)
        self.nlevel = len(self.data_cal['x'].unique())
    def drop_obs(self,obs):
        self.data_cal = self.data_cal.drop(obs)
        self.nobs = len(self.data_cal.index)
        self.nlevel = len(self.data_cal['x'].unique())

class batch():
    def __init__(self,data_cal = 0,data_sample = 0,data_val = 0,MultiIndex = True,index_var = 'x'):
        self.data_cal = data_cal
        self.data_sample = data_sample
        self.data_val = data_val
        self.list = []
        df = self.data_cal
        if MultiIndex == True:
            x = [float(i) for i in list(zip(*df.index))[0]]
        else:
            x = [float(i) for i in df.index]
        index = df.index
        for i in range(len(df.columns)):
            name = df.columns[i]
            y = df.iloc[:,i].values
            score = np.zeros(len(y))
            weight = np.ones(len(y))
            if index_var == 'x':
                dfa = pd.DataFrame({'x': x,'y': y,'select': score, 'accuracy': score,'weight': weight},index = index)
            elif index_var == 'y':
                dfa = pd.DataFrame({'x': y,'y': x,'select': score, 'accuracy': score,'weight': weight},index = index)
            self.list.append(calibration(dfa,name))
    def add_sample(self,sample):
        for i in range(len(self.list)):
            index = sample.index
            y = sample.iloc[:,i].values
            x = np.zeros(len(y))
            dfb = pd.DataFrame({'x': x,'y': y},index = index) 
            self.list[i].data_sample = dfb
        self.data_sample = sample
    def quantify(self,rangelimit = False):
        self.results = self.data_sample.copy()
        if rangelimit == True:
            for i,j in enumerate(self.list):
                j.quantify(rangelimit = True)
                self.results.iloc[:,i] = j.data_sample['x']
        else:
            for i,j in enumerate(self.list):
                j.quantify(rangelimit = False)
                self.results.iloc[:,i] = j.data_sample['x']
    def fit(self,optimize = True,target = "accuracy", lloq_crit = 0.2,crit = 0.15, order = 'weight' , rangemode = 'auto', selectmode = 'hloq stepdown',
            lloq = 0, hloq = -1, weights = ['1','1/x^0.5','1/x','1/x^2'], fixpoints = [],model = lm,repeat_weight = True):
        for i in self.list:
            i.fit(optimize=optimize,target=target,lloq_crit=lloq_crit,crit=crit,order=order,rangemode=rangemode,selectmode=selectmode, 
                      lloq=lloq, hloq=hloq, weights=weights, fixpoints=fixpoints,model=model,repeat_weight=repeat_weight)
    def plot(self,n=2,neg = False,ylabel = 0, xlabel = 0):
        plot_calibration_line(batch = self,n=n,neg = neg,ylabel = ylabel, xlabel = xlabel)

def build_multical(self):
    def __intit__(self):
        self.multical = []
        df = self.data_cal
        xname = list(zip(*df.index))[0]
        x = list(zip(*df.index))[1]
        index = df.index
        for i in range(len(df.columns)):
            yname = df.columns[i]
            for j in xname.unique():
                y = df.iloc[:,i].loc[j].values
                x = list(zip(*df.loc[j].index))[1]
                score = np.zeros(len(y))
                weight = np.ones(len(y))
                dfa = pd.DataFrame({'x': x,'y': y,'select': score, 'accuracy': score,'weight': weight},index = index)
