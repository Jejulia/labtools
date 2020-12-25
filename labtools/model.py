import pandas as pd
import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
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
        self.nobs = len(self.x)
    def fit(self):
        self.X = sm.add_constant(self.x)
        w = np.sqrt(self.w)
        X = w[:, None] * self.X
        Y = w[:, None] * self.y[:,None]
        F = np.dot(X.T,X)
        U = sl.cholesky(F)
        z = sl.solve(U.T,np.dot(X.T,Y))
        self.beta = sl.solve(U,z).ravel()
        yhat = self.predict(self.x)
        self.rsquared = 1 - ((w * yhat - Y.ravel())**2).sum()/(self.w * (self.y - np.average(self.y, weights = self.w))**2).sum()
    def predict(self,x):
        if x.shape[0] == 1:
            x = np.array([1, x[0]])
        else:
            x = sm.add_constant(x)
        return np.dot(x,self.beta)
    def invpred(self,y):
        return (y-self.beta[0])/self.beta[1]
    def score(self,target):
        return target(self.x,self.y,self.beta)

### target functions
def xaccuracy(x,y,params):
    return (y-params[0])/params[1]/x
def xbias(x,y,params):
    return xaccuracy(x,y,params)-1
def xresidual(x,y,params):
    return (y-params[0])/params[1]-x
def yaccuracy(x,y,params):
    return (params[1]*x+params[0])/y
def ybias(x,y,params):
    return yaccuracy(x,y,params)-1
def yresidual(x,y,params):
    return (params[1]*x+params[0])-y
    
class calibration():
    def __init__(self,df,name,sample = None):
        self.data_cal = df
        self.data_sample = sample
        self.name = name
        self.nobs = len(self.data_cal.index)
        self.nlevel = len(self.data_cal['x'].unique())
    def accuracy(self,x,y,params):
        return (y-params[0])/params[1]/x
    def fit(self, model = lm, target = xbias, crit = 0.15, lloq_crit = 0.2, **kwargs):
        # avalible order: 'range', 'weight'
        # available rangemode: 'auto', 'unlimited'
        # available weights: '1','1/x^0.5','1/x','1/x^2','1/y^0.5','1/y','1/y^2'
        # available selectmode: 'hloq stepdown','lloq stepup','sequantial stepdownup','sequantial stepupdown'
        kwdict = dict(order = 'weight' , rangemode = 'auto', selectmode = 'hloq stepdown',
            lloq = 0, hloq = -1, weights = ['1','1/x^0.5','1/x','1/x^2'], fixpoints = [],repeat_weight = True)
        for k,v in kwargs.items():
            kwdict[k] = v
        om = op.linear(self.data_cal, self.data_sample, target, crit, lloq_crit, **kwdict)
        print(self.name)
        om.fit(model)
        self.model = om.model
        self.data_cal['select'] = om.select_final
        self.data_cal['weight'] = om.weight_type[om.weight](om.x,om.y)
        self.weight = om.weight
        self.data_cal['accuracy'] = self.accuracy(om.x,om.y,self.model.beta)
        print('done')
        del om
    def quantify(self,limit=False):
        self.data_sample['x'] = self.model.invpred(self.data_sample['y'])
        if limit:
            for conc in self.data_sample['x']:
                if conc < min(self.model.x):
                    self.data_sample.loc[self.data_sample['x'] == conc,'x'] = '< LLOQ'
                elif conc > max(self.model.x):
                    self.data_sample.loc[self.data_sample['x'] == conc,'x'] = '> HLOQ'
    def drop_level(self,level):
        self.data_cal = self.data_cal.drop(level)
        self.nobs = len(self.data_cal.index)
        self.nlevel = len(self.data_cal['x'].unique())
    def drop_obs(self,obs):
        self.data_cal = self.data_cal.iloc[[i for i in range(self.nobs) if i not in obs],:]
        self.nobs = len(self.data_cal.index)
        self.nlevel = len(self.data_cal['x'].unique())

class batch():
    def __init__(self,data_cal = 0,data_sample = 0,data_val = 0,MultiIndex = True,index_var = 'x'):
        self.data_cal = data_cal
        self.data_sample = data_sample
        self.data_val = data_val
        self.calibration = []
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
            self.calibration.append(calibration(dfa,name))
    def add_sample(self,sample):
        for i in range(len(self.calibration)):
            index = sample.index
            y = sample.iloc[:,i].values
            x = np.zeros(len(y))
            dfb = pd.DataFrame({'x': x,'y': y},index = index) 
            self.calibration[i].data_sample = dfb
        self.data_sample = sample
    def quantify(self,limit=False):
        self.results = self.data_sample.copy()
        for ind,analyte in enumerate(self.calibration):
            analyte.quantify(limit)
            self.results.iloc[:,ind] = analyte.data_sample['x']
    def fit(self, model = lm, target = xbias, crit = 0.15, lloq_crit = 0.2, order = 'weight' , rangemode = 'auto', selectmode = 'hloq stepdown',
            lloq = 0, hloq = -1, weights = ['1','1/x^0.5','1/x','1/x^2'], fixpoints = [],repeat_weight = True):
        for analyte in self.calibration:
            analyte.fit(model, target, crit, lloq_crit, order=order,rangemode=rangemode,selectmode=selectmode, 
                      lloq=lloq, hloq=hloq, weights=weights, fixpoints=fixpoints,repeat_weight=repeat_weight)
    def plot(self,n=2,ylabel = 0, xlabel = 0):
        plot_calibration_line(batch = self,n=n,ylabel = ylabel, xlabel = xlabel)

