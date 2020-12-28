from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np
import pandas as pd
import argparse

def tidy(file, parameter='ISTD Resp. Ratio', cal=False):
    # available parameter: 'ISTD Resp. Ratio', 'Area', 'Final Conc.', 'Resp.', etc
    # cal = True, to calculate calibration points 
    format = file.split("\\")[-1].split('.')[-1]
    if format == 'csv':
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    if cal == False:
        df = df.loc[df['Unnamed: 4'] != 'Cal']
    
    
    # set index, drop useless columns
    df = df.set_index(['Unnamed: 2', 'Unnamed: 3'])._set_axis_name(['Sample', 'Data File'])
    
    # format analyte name
    Analyte = list(df.columns)
    analyte = '(ISTD)'
    for ind,name in enumerate(Analyte):
        if 'Unnamed' in name:
            Analyte[ind] = analyte
        else:
            analyte = Analyte[ind] = name[0:-8] # ret rid of ' Results', get rid of 'Unnamed: '

    # drop ISTD, select 'ISTD Resp. Ratio'/'Area'/'Conc.'/'Resp.'
    select = []
    ind = 0
    for analyte, parameters in zip(Analyte, df.iloc[0,:]):
        if ('(ISTD)' not in analyte) and (parameters == parameter):
            select.append(ind)
        ind += 1
    #  drop the first row, set column, select, turn data type into float
    df = df.set_axis(Analyte, axis=1).iloc[1:, select].astype(float) 
    
    return df

# default stats method
def __mean__(df):
    return df.mean()

def __sd__(df):
    return df.std()

def __rsd__(df):
    return __sd__(df)/__mean__(df)



def stats(df, Stats = ['mean', 'sd', 'rsd']):
    """
    stats(df,Stats = ['mean', 'sd', 'rsd'])

    Summary simple stats of multiple samples.

    Parameters
    ----------
    df : pandas.DataFrame
        The index should be MultiIndex with two levels. The first level is sample ID, the second level is repeated measures of samples.
    Stats : array_like
        An array of strings that indicates which stats are to be calculated. The correspond function names are uppercased of the elements.
    
    Returns
    -------
    out : pandas.DataFrame

    Examples
    --------
    >>> df = pd.DataFrame({'ID':np.repeat(['a','b'],2),'measure':[1,2,1,2],
                    'A':list(range(4)),'B':list(range(1,5))}).set_index(['ID','measure'])
    >>> df
                 A  B
    ID  measure  
     a        1  0  1
     a        2  1  2
     b        1  2  3
     b        2  3  4

    >>> lbt.stats(df)
             Analyte         A         B                          
    Samples    Stats
          a     mean  0.500000  1.500000     
                  sd  0.707107  0.707107
                 rsd  1.414214  0.471405
          b     mean  2.500000  3.500000
                  sd  0.707107  0.707107
                 rsd  0.282843  0.202031


    
    User defined stats:

    >>> def rsd_per(df):
            return df.std()/df.mean() * 100
    >>> lbt.__setattr__('__rsd_per__', rsd_per)
    >>> lbt.stats(df, ['mean', 'sd', 'rsd_per'])
             Analyte           A          B                          
    Samples    Stats
          a     mean    0.500000   1.500000     
                  sd    0.707107   0.707107
             rsd_per  141.421356  47.140452
          b     mean    2.500000   3.500000
                  sd    0.707107   0.707107
             rsd_per   28.284271  20.203051

    """
    samples, _ = zip(*list(df.index))
    samples = pd.Series(samples).unique()
    # Use preallocated dataframe instead?
    row_list = []
    for sample in samples:
        for stats in Stats:
            row_list.append(eval('__{}__'.format(stats))(df.loc[sample]))
            #exec('{} = __{}__(df.loc[sample])'.format(stats, stats))
            #exec('row_list.append({})'.format(stats))
    index1 = np.repeat(samples, len(Stats))
    index2 = np.repeat([Stats], len(samples), axis=0).ravel()
    dfstats = pd.DataFrame(row_list).set_index([index1, index2]).set_axis(df.columns, axis=1)._set_axis_name(['Samples','Stats'])._set_axis_name(['Analyte'], axis=1)
    return dfstats

def savedf(df):
    save = asksaveasfilename()
    format = save.split("\\")[-1].split('.')[-1]
    if format == 'csv':
        df.to_csv(save)
    elif format == 'xlsx' or format == 'xls':
        df.to_excel(save)
    elif format == 'json':
        df.to_json(save)
    elif format == 'html':
        df.to_html(save)


