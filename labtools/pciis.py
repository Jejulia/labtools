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

global data 
data = {}


def quantify(chrom,trans,save=False):
    
    # import data
    
    try:
        df_trans = data[trans]
    except:
        df_trans = pd.read_excel(trans)
        data[trans] = df_trans
    if chrom.split(".")[-1] == 'xls':
        from pyexcel_xls import read_data
    else:
        from pyexcel_xlsx import read_data
    try:
        df_chrom = data[chrom]
    except:
        df_chrom = read_data(chrom)
        data[chrom] = df_chrom
        
    # pci transition, retention time
    
    df_trans['Transition'] = [re.sub("^ *","",i) for i in df_trans['Transition']]
    # pci_name = re.sub("[(]PCI[)]","",df_trans.iloc[['(PCI)' in i for i in df_trans.iloc[:,0]],0][0])
    pci_trans = df_trans.loc[['(PCI)' in i for i in df_trans.iloc[:,0]],'Transition'][0]
    df_trans = df_trans.set_index(['Transition'])
    rt_min = min(df_trans['RT.s'].dropna())
    rt_max = max(df_trans['RT.e'].dropna())

    rand = random.sample(list(df_chrom.keys()),1)
    df = np.vstack(df_chrom[rand[0]][2:])
    df = df[(np.searchsorted(df[:,1],rt_min,side='right')-1):(np.searchsorted(df[:,1],rt_max)+1),:]
    iv = pd.Series([df[i+1,1]-df[i,1] for i in range(df.shape[0]-1)])
    iv = round(iv.mode()[0],4)
    rt = np.arange(df[0,1],df[-1,1]+iv, iv)
    
    # filter chromatography
    
    result = [interpolate1(df,rt) for df in df_chrom.values()]
    mat_chrom = np.vstack(list(zip(*result))[0]).transpose() # rt x trans
    name = list(zip(*result))[1]

    # Calculate ratio
    
    datafile = [re.sub('.*[(]',"",i) for i in name]
    trans = pd.Series([re.sub('[)].*',"",i) for i in datafile])
    datafile = [re.sub('.*[) ]',"",i) for i in datafile]
    pci_index = trans == pci_trans
    mat_pci = mat_chrom[:,pci_index]
    ntrans = len(pci_index)
    for ind in range(mat_chrom.shape[1]//ntrans):
        mat_chrom[:,range(ntrans*ind,ntrans*ind+ntrans)] = mat_chrom[:,range(ntrans*ind,ntrans*ind+ntrans)]/mat_pci
    
    # Peak computing

    dict_range = dict()
    for tran in df_trans.index:
        if tran == pci_trans:
            dict_range[tran] = list(range(len(rt)))
        else:
            dict_range[tran] = [ind for ind,time in enumerate(rt) if time > df_trans.loc[tran,'RT.s'] and time < df_trans.loc[tran,'RT.e']]
            
    mat_chrom = np.array([sum(mat_chrom[dict_range[trans[ind]],ind])-sum(mat_chrom[dict_range[trans[ind]][[0,-1]],ind][[0,-1]])/2 for ind in range(len(datafile))])
    
    # data assembly
    
    datafile = pd.Series(datafile).unique()
    trans = trans.unique()
    peak = mat_chrom.reshape(len(trans),len(datafile)).transpose()
    peak = pd.DataFrame(peak)
    peak.index = datafile
    peak.columns = trans
    
    # Save
    
    if save == True:
        try:
            peak.to_excel('{}.xlsx'.format(asksaveasfilename()))
        except:
            "Cancelled"
    return peak


def ionsuppression(chrom,trans,window = 10,save=False,data = data):
    
    # import data
    
    try:
        df_trans = data[trans]
    except:
        df_trans = pd.read_excel(trans)
        data[trans] = df_trans
    if chrom.split(".")[-1] == 'xls':
        from pyexcel_xls import read_data
    else:
        from pyexcel_xlsx import read_data
    try:
        df_chrom = data[chrom]
    except:
        df_chrom = read_data(chrom)
        data[chrom] = df_chrom
    
    # pci transition, retention time
    
    df_trans['Transition'] = [re.sub("^ *","",i) for i in df_trans['Transition']]
    # pci_name = re.sub("[(]PCI[)]","",df_trans.iloc[['(PCI)' in i for i in df_trans.iloc[:,0]],0][0])
    pci_trans = df_trans.loc[['(PCI)' in i for i in df_trans.iloc[:,0]],'Transition'][0]
    
    while True:
        rand = random.sample(list(df_chrom.keys()),1)
        if pci_trans in df_chrom[rand[0]][0][0]:
            df = np.vstack(df_chrom[rand[0]][2:])
            break
    df = df[df[:,1]<2,:]
    iv = pd.Series([df[i+1,1]-df[i,1] for i in range(df.shape[0]-1)])
    iv = round(iv.mode()[0],4)
    argmin = df[:,2].argmin()
    rt = np.arange(df[argmin,1]-window*iv,df[argmin,1]+(window+1)*iv, iv)
    
    # filter chromatography
    
    result = np.array([interpolate2(df,rt,pci_trans) for df in df_chrom.values()])
    result = result[result.nonzero()]
    mat_chrom = np.vstack(list(zip(*result))[0])
    name = list(zip(*result))[1]
    
    # data assembly
    
    datafile = [re.sub('.*[(]',"",re.sub('.*[) ]',"",i)) for i in name]
    df_is = pd.DataFrame(mat_chrom)
    df_is.columns = rt
    df_is.index = pd.Series(datafile).unique()
    
    # Save
    
    if save == True:
        try:
            df_is.to_excel('{}.xlsx'.format(asksaveasfilename()))
        except:
            "Cancelled"
    return df_is


def sumif(mat,i,trans,dict_range):
    return sum(mat[dict_range[trans[i]],i])


def interpolate1(df,rt):
    name = df[0][0]
    df = np.vstack(df[2:])
    return  [np.interp(rt,df[:,1],df[:,2]),name]


def interpolate2(df,rt,pci_trans):
    name = df[0][0]
    if pci_trans not in name:
        pass
    else:
        df = np.vstack(df[2:])
        return  [np.interp(rt,df[:,1],df[:,2]),name]
