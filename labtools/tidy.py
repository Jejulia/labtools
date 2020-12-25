from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np
import pandas as pd

def tidy(file,parameter='ISTD Resp. Ratio',cal=False):
    # available parameter: 'ISTD Resp. Ratio', 'Area', 'Final Conc.', 'Resp.', etc
    # cal = True, to calculate calibration points
    format = file.split("\\")[-1].split('.')[-1]
    if format == 'csv':
        rawdf = pd.read_csv(file)
    else:
        rawdf = pd.read_excel(file)
    if cal == False:
        rawdf = rawdf.loc[rawdf['Unnamed: 4'] != 'Cal']
    Sample = pd.Index(rawdf.iloc[1:,2],name = 'Sample')
    Data_file = pd.Index(rawdf.iloc[1:,3],name = 'Data file')
    Analyte = list(rawdf.columns)[7:]
    for i,j in enumerate(Analyte):
        if 'Unnamed' in j:
            Analyte[i] = j
        else:
            Analyte[i] = j[0:-8] # ret rid of ' Results'
    for i,j in enumerate(Analyte):
        if 'Unnamed' in j:
            Analyte[i] = k
        else:
           k =  Analyte[i] # get rid of 'Unnamed: '
    Analyte = pd.Index(Analyte,name = 'Analyte')
    Parameter = pd.Index(rawdf.iloc[0,7:],name = 'Parameter')
    revisedf = rawdf.drop(rawdf.iloc[:,0:7],axis = 1).drop(0) # drop the useless first 8 columns and the first row
    revisedf = revisedf.set_index([Sample,Data_file])
    revisedf = revisedf.set_axis([Analyte,Parameter],axis = 1)
    revisedf = revisedf.dropna(axis = 0) # drop useless parameters, the value would be na
    select = [i for i,j in enumerate([i for i,j in revisedf.columns]) if '(ISTD)' not in j] # drop ISTD
    finaldf = revisedf.iloc[:,select]
    finaldf = finaldf.stack(level = 0)
 
    finaldf = finaldf[parameter].unstack().reindex(Sample.unique(),level=0).astype(float) # select 'ISTD Resp. Ratio'/'Area'/'Conc.'/'Resp.'
    
    return finaldf

def stat(df):
    Stats = ['mean','sd','rsd']
    statlen = len(Stats)
    sample,datafile = zip(*list(df.index))
    sample = pd.Series(sample).unique()
    for i in range(len(sample)):
        Sample = np.repeat(sample[i],statlen)
        index = pd.MultiIndex.from_arrays([Sample,Stats],names = ['Level','Stats'])
        dfsub = pd.DataFrame([df.loc[sample[i]].mean(),df.loc[sample[i]].std(),df.loc[sample[i]].std()/df.loc[sample[i]].mean()],index = index).transpose()
        if i == 0:
            dfstats = dfsub
        else:
            dfstats = dfstats.join(dfsub)        
    index = pd.MultiIndex.from_arrays([[dfstats.index[int(i/statlen)] for i in range(len(dfstats.index)*3)],[Stats[i%statlen] for i in range(len(dfstats.index)*statlen)]],names = ['Analyte','Stats'])
    dfstats = dfstats.transpose().unstack(1).reindex(columns=index)
    return dfstats

def savedf(df,parameter,stats = False):
    save = asksaveasfilename()
    if stats == False:
        df.to_excel(save+'({},data).xlsx'.format(parameter))
    else:
        df.to_excel(save+'({},stats).xlsx'.format(parameter))

