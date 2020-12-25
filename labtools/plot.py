from scipy.stats import ttest_ind, ttest_rel, median_test, wilcoxon, bartlett, levene, fligner
from scipy.stats import f as ftest
from scipy.stats.mstats import mannwhitneyu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def plot_calibration_line(batch,n=2,neg = False,ylabel = 0, xlabel = 0):
    for i in range(len(batch.list)):
        analyte = batch.list[i]
        fig, ax = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.figsize'] = [6*n,4*n]
        plt.rcParams['font.size'] = 10*n
        ax.scatter(analyte.data_cal['x'],analyte.data_cal['y'],s = 15*n,c = 'r',alpha = 0.7,marker = 'x')
        lin = np.linspace(min(analyte.model.x),max(analyte.model.x),1000)
        lin2 = analyte.model.fit.predict(sm.add_constant(lin))
        ax.plot(lin,lin2,c = 'chartreuse',lw = 2*n,alpha = 0.7)
        ax.scatter(analyte.model.x,analyte.model.y,c = 'b',s = 15*n)
        plt.xlim([min(lin)-(max(lin)-min(lin))*0.1,max(lin)+(max(lin)-min(lin))*0.1])
        plt.ylim([min(lin2)-(max(lin2)-min(lin2))*0.1,max(lin2)+(max(lin2)-min(lin2))*0.1])
        
        if xlabel == 0:
            ax.set_xlabel('{} concentration (ng/mL)'.format(analyte.name))
        else:
            ax.set_xlabel(xlabel[i])
        if ylabel == 0:
            ax.set_ylabel('Corrected signal')
        else:
            ax.set_ylabel(ylabel[i])
        if not neg:
            ax.annotate(r'$y = {}+{}x$'.format(round(analyte.model.fit.params[0],3),round(analyte.model.fit.params[1],3)),
                        xy = ((max(lin)+3*min(lin))/4,(3*max(analyte.model.y)+min(analyte.model.y))/4))
            ax.annotate(r'$r^2 = {}$'.format(round(analyte.model.fit.rsquared,3)),
                        xy = ((max(lin)+3*min(lin))/4,(2.7*max(analyte.model.y)+1.3*min(analyte.model.y))/4))
        else:
            ax.annotate(r'$y = {}+{}x$'.format(round(analyte.model.fit.params[0],3),round(analyte.model.fit.params[1],3)),
                        xy = ((3*max(lin)+min(lin))/4,(3*max(analyte.model.y)+min(analyte.model.y))/4))
            ax.annotate(r'$r^2 = {}$'.format(round(analyte.model.fit.rsquared,3)),
                        xy = ((3*max(lin)+min(lin))/4,(2.7*max(analyte.model.y)+1.3*min(analyte.model.y))/4))
        plt.tight_layout()
        plt.show()


def starplot(df = [],x = '',y = '',data = [],index = [],columns = [],
             fold = False,foldcol = 0,mode = 3, errorbar = True, 
             plottype = 'barplot', stats = 'independent t test',
             test_var = False, stats_var = 'f test', crit_var = 0.05, equal_var = True,
             rotate = 0, elinewidth = 0.5, fontsize = 14, capsize = 4,
             noffset_ylim = 35, noffset_fst = 10,noffset_diff = 10,star_size = 3,linewidth = 1,
             crit = [0.05,0.01,0.001,0.0001]):
    # data: list of data matrixs(or DataFrames) for comparison (row: obs, columns: var)
    # index: var, columns: obs
    # adjacent: annotate star for adjacent bar
    # control: annotate star between all other bars to selctive control bar
    # mix: mix mode
    # 3: annotate star for all combination of bar (only 3 bars available)

    crit = np.array(crit)
    plt.rcParams['font.family'] = 'Times New Roman'
    fig,ax = plt.subplots()
    star = ['*','**','***','****']
    n = len(data)
    m = data[0].shape[1]
    test = pd.DataFrame()
    for i,j in enumerate(data):
        if type(test) == type(j):
            data[i] = j.values.reshape(len(j.index),len(j.columns))
    if plottype == 'barplot':
        error = pd.DataFrame()
        mean = pd.DataFrame()
        for i in range(m):
            error[i] = [data[j][:,i].std() for j in range(n)]
            mean[i] = [data[j][:,i].mean() for j in range(n)]
        error = error.transpose()
        mean = mean.transpose()
        if len(index) != 0:
            error.index = index
            mean.index = index
        if len(columns) != 0:
            error.columns = columns
            mean.columns = columns
        if fold == True:
            oldmean = mean.copy()
            olderror = error.copy()
            for i in range(len(mean.columns)):
                mean.iloc[:,i] = oldmean.iloc[:,i]/oldmean.iloc[:,foldcol]
                error.iloc[:,i] = olderror.iloc[:,i]/oldmean.iloc[:,foldcol]
        if errorbar == True:
            plot = plot = mean.plot.bar(yerr = error,ax = ax,rot=rotate,capsize = capsize, 
                                        error_kw=dict(elinewidth = elinewidth),fontsize = fontsize)
            max_bar = [[mean.iloc[j,i]+error.iloc[j,i] for i in range(n)] for j in range(m)]
            min_bar = [mean.iloc[j,i]-error.iloc[j,i] for i in range(n) for j in range(m)]
        else:
            plot = plot = mean.plot.bar(ax = ax,rot=rotate,capsize = capsize, 
                                        error_kw=dict(elinewidth = elinewidth),fontsize = fontsize)
            max_bar = [[mean.iloc[j,i] for i in range(n)] for j in range(m)]
            min_bar = [mean.iloc[j,i] for i in range(n) for j in range(m)]
    elif plottype == 'boxplot':
        print("under buiding")
    ylim = 0
    offset = max([max_bar[i][j] for i in range(m) for j in range(n)])/100
    blank = []
    if mode == 3:
        for j in range(m):
            level = np.zeros(n)
            for i in range(n):
                if i < n-1:
                    k = i+1
                else:
                    k = 0
                if test_var == True:
                    if stats_var == 'f test':
                        f = 0.5-abs(0.5-ftest.sf(data[i][:,j].var()/data[k][:,j].var(),len(data[i][:,j])-1,len(data[k][:,j])-1))
                        if crit_var/2 > f:
                            equal_var = False
                        else:
                            equal_var = True
                    else:
                        if stats_var == 'bartlett':
                            f = bartlett(data[i][:,j],data[k][:,j])[1]
                        elif stats_var == 'levene':
                            f = bartlett(data[i][:,j],data[k][:,j])[1]
                        elif stats_var == 'fligner':
                            f = fligner(data[i][:,j],data[k][:,j])[1]
                        if crit_var > f:
                            equal_var = False
                        else:
                            equal_var = True
                if stats == 'independent t test':
                    p = ttest_ind(data[i][:,j],data[k][:,j],equal_var = equal_var)[1]
                elif stats == 'paired t test':
                    if equal_var == True:
                        p = ttest_rel(data[i][:,j],data[k][:,j])[1]
                    else:
                        p = 0
                elif stats == 'median test':
                    p = median_test(data[i][:,j],data[k][:,j])[1]
                elif stats == 'mannwhitneyu':
                    if equal_var == True:
                        p = mannwhitneyu(data[i][:,j],data[k][:,j])[1]
                    else:
                        p = 0
                elif stats == 'wilcoxon':
                    if equal_var == True:
                        p = wilcoxon(data[i][:,j],data[k][:,j])[1]
                    else:
                        p = 0
                level[i] = len(crit) - len(crit.compress(p > crit))
            for k in range(n):
                height = 0
                if level[k] != 0 and k != n-1:
                    center = [plot.patches[k*m+j].get_x(), plot.patches[k*m+m+j].get_x()]
                    height = max([max_bar[j][k],max_bar[j][k+1]])
                    h1 = max_bar[j][k]
                    h2 = max_bar[j][k+1]
                    width = plot.patches[k*m+j].get_width()
                    blank.append((center[0]+width/2,height+noffset_fst*offset+(-1)**k*2*offset))
                    blank.append((center[1]+width/2,height+noffset_fst*offset+(-1)**k*2*offset))
                    ax.vlines(x = center[0]+width/2,ymin = h1+offset*2,ymax = height+noffset_fst*offset+(-1)**k*2*offset,lw = linewidth)
                    ax.vlines(x = center[1]+width/2,ymin = h2+offset*2,ymax = height+noffset_fst*offset+(-1)**k*2*offset,lw = linewidth)
                    ax.annotate(star[int(level[k]-1)],
                                xy = ((center[0]+center[1])/2+width/2,height+(noffset_fst+1)*offset+(-1)**k*2*offset),
                                ha='center',size = star_size)
                elif level[k] != 0 and k == n-1:
                    center = [plot.patches[j].get_x(), plot.patches[k*m+j].get_x()]
                    height = max(max_bar[j])
                    h1 = max_bar[j][0]
                    h2 = max_bar[j][k]
                    blank.append((center[0]+width/2,height+(noffset_fst+noffset_diff)*offset))
                    blank.append((center[1]+width/2,height+20*offset))
                    ax.vlines(x = center[0]+width/2,ymin = h1+offset*2,ymax = height+(noffset_fst+noffset_diff)*offset,lw = linewidth)
                    ax.vlines(x = center[1]+width/2,ymin = h2+offset*2,ymax = height+(noffset_fst+noffset_diff)*offset,lw = linewidth)
                    ax.annotate(star[int(level[k]-1)],
                                xy = ((center[0]+center[1])/2+width/2,height+(noffset_fst+noffset_diff+1)*offset),
                                ha='center',size = star_size)
                if height > ylim:
                    ylim = height
    if mode == 'adjacent':
        for j in range(m):
            level = np.zeros(n-1)
            for i in range(n-1):
                k = i+1
                if test_var == True:
                    if stats_var == 'f test':
                        f = 0.5-abs(0.5-ftest.sf(data[i][:,j].var()/data[k][:,j].var(),len(data[i][:,j])-1,len(data[k][:,j])-1))
                        if crit_var/2 > f:
                            equal_var = False
                        else:
                            equal_var = True
                    else:
                        if stats_var == 'bartlett':
                            f = bartlett(data[i][:,j],data[k][:,j])[1]
                        elif stats_var == 'levene':
                            f = bartlett(data[i][:,j],data[k][:,j])[1]
                        elif stats_var == 'fligner':
                            f = fligner(data[i][:,j],data[k][:,j])[1]
                        if crit_var > f:
                            equal_var = False
                        else:
                            equal_var = True
                if stats == 'independent t test':
                    p = ttest_ind(data[i][:,j],data[k][:,j],equal_var = equal_var)[1]
                elif stats == 'paired t test':
                    if equal_var == True:
                        p = ttest_rel(data[i][:,j],data[k][:,j])[1]
                    else:
                        p = 0
                elif stats == 'median test':
                    p = median_test(data[i][:,j],data[k][:,j])[1]
                elif stats == 'mannwhitneyu':
                    if equal_var == True:
                        p = mannwhitneyu(data[i][:,j],data[k][:,j])[1]
                    else:
                        p = 0
                elif stats == 'wilcoxon':
                    if equal_var == True:
                        p = wilcoxon(data[i][:,j],data[k][:,j])[1]
                    else:
                        p = 0
                level[i] = len(crit) - len(crit.compress(p > crit))
            for k in range(n-1):
                height = 0
                if level[k] != 0:
                    center = [plot.patches[k*m+j].get_x(), plot.patches[k*m+m+j].get_x()]
                    height = max([max_bar[j][k],max_bar[j][k+1]])
                    h1 = max_bar[j][k]
                    h2 = max_bar[j][k+1]
                    width = plot.patches[k*m+j].get_width()
                    blank.append((center[0]+width/2,height+noffset_fst*offset+(-1)**k*2*offset))
                    blank.append((center[1]+width/2,height+noffset_fst*offset+(-1)**k*2*offset))
                    ax.vlines(x = center[0]+width/2,ymin = h1+offset*2,ymax = height+noffset_fst*offset+(-1)**k*2*offset,lw = linewidth)
                    ax.vlines(x = center[1]+width/2,ymin = h2+offset*2,ymax = height+noffset_fst*offset+(-1)**k*2*offset,lw = linewidth)
                    ax.annotate(star[int(level[k]-1)],
                                xy = ((center[0]+center[1])/2+width/2,height+(noffset_fst+1)*offset+(-1)**k*2*offset),
                                ha='center',size = star_size)
                if height > ylim:
                    ylim = height
    ax.set_ylim(min(0,min(min_bar)-10*offset),ylim+noffset_ylim*offset)
    for j,i in enumerate(blank):
        ax.vlines(x = i[0], ymin = i[1],ymax = i[1]+offset*2,color = 'white',lw = 1.2*linewidth)
        if j%2 == 1:
            ax.hlines(y = i[1], xmin = blank[j-1], xmax = blank[j],lw = linewidth)
