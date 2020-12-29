from scipy.stats import ttest_ind, ttest_rel, median_test, wilcoxon, bartlett, levene, fligner
from scipy.stats import f as ftest
from scipy.stats.mstats import mannwhitneyu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def plot_calibration_line(batch, n = 2, ylabel = 0, xlabel = 0):
    for i in range(len(batch.calibration)):
        # settings
        fig, ax = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.figsize'] = [6*n, 4*n]
        plt.rcParams['font.size'] = 10*n

        # plot 
        ax.scatter(batch.calibration[i].data_cal['x'], batch.calibration[i].data_cal['y'], s = 15*n, c = 'r', alpha = 0.7, marker = 'x')
        lin = np.linspace(min(batch.calibration[i].model.x), max(batch.calibration[i].model.x), 1000)
        lin2 = batch.calibration[i].model.predict(sm.add_constant(lin))
        ax.plot(lin, lin2, c = 'chartreuse', lw = 2*n, alpha = 0.7)
        ax.scatter(batch.calibration[i].model.x, batch.calibration[i].model.y, c = 'b', s = 15*n)
        plt.xlim([min(lin)-(max(lin)-min(lin))*0.1, max(lin)+(max(lin)-min(lin))*0.1])
        plt.ylim([min(lin2)-(max(lin2)-min(lin2))*0.1, max(lin2)+(max(lin2)-min(lin2))*0.1])
        
        # x,y labels
        if xlabel == 0:
            ax.set_xlabel('{} concentration (ng/mL)'.format(batch.calibration[i].name))
        else:
            ax.set_xlabel(xlabel[i])
        if ylabel == 0:
            ax.set_ylabel('Corrected signal')
        else:
            ax.set_ylabel(ylabel[i])
        
        # parameter labels
        r2 = r'$r^2 = %s$'%(round(batch.calibration[i].model.rsquared, 3))
        weight = batch.calibration[i].weight ###### To be modified: weight
        if '^' in weight:
            left,right = weight.split('^')
            if right == '0.5':
                weight = r'$weight: %s/\sqrt{%s}$'%(left.split('/')[0], left.split('/')[1])
            else:
                weight = r'$weight: %s/%s^%s$'%(left.split('/')[0], left.split('/')[1], right)
        else:
            weight = r'$weight: %s$'%(weight)
        x1 = (max(lin)+3*min(lin))/4
        x2 = (3*max(lin)+min(lin))/4
        y1 = (3*max(batch.calibration[i].model.y)+min(batch.calibration[i].model.y))/4
        y2 = (2.7*max(batch.calibration[i].model.y)+1.3*min(batch.calibration[i].model.y))/4
        y3 = (2.4*max(batch.calibration[i].model.y)+1.6*min(batch.calibration[i].model.y))/4
        if batch.calibration[i].model.beta[1] > 0:
            equation = r'$y = %s+%sx$'%(round(batch.calibration[i].model.beta[0], 3), round(batch.calibration[i].model.beta[1], 3))
            ax.annotate(equation,xy = (x1,y1))
            ax.annotate(r2,xy = (x1,y2))
            ax.annotate(weight,xy = (x1,y3))
        else:
            equation = r'$y = %s%sx$'%(round(batch.calibration[i].model.beta[0], 3), round(batch.calibration[i].model.beta[1], 3))
            ax.annotate(equation, xy = (x2, y1))
            ax.annotate(r2, xy = (x2, y2))
            ax.annotate(weight, xy = (x2, y3))
        plt.tight_layout()
        plt.show()

def _test_var(stats, crit, group1, group2):
    if stats == 'f test':
        f = 0.5-abs(0.5-ftest.sf(group1.var()/group2.var(), group1.shape[0]-1, group2.shape[0]-1))
        return [crit/2 < qf for qf in f]
    else:
        if stats == 'bartlett':
            return [crit_var < bartlett(group1.iloc[:, ifeature],group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]
        elif stats == 'levene':
            return [crit < levene(group1.iloc[:, ifeature],group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]
        elif stats == 'fligner':
            return [crit < fligner(group1.iloc[:, ifeature],group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]

def _test_cen(stats, crit, equal_var, group1, group2):
    crit = np.array(crit)
    if stats == 'independent t test':
        p = [ttest_ind(group1.iloc[:, ifeature], group2.iloc[:, ifeature], equal_var = equal_var[ifeature])[1] for ifeature in range(group1.shape[1])]
    elif stats == 'paired t test':
        p = ttest_rel(group1, group2)[1]
    elif stats == 'median test':
        p = [median_test(group1.iloc[:, ifeature], group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]
    elif stats == 'mannwhitneyu':
        p = [mannwhitneyu(group1.iloc[:, ifeature], group2.iloc[:,ifeature])[1] for ifeature in range(group1.shape[1])]
    elif stats == 'wilcoxon':
        p = [wilcoxon(group1.iloc[:, ifeature], group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]

    return [len(crit) - len(crit.compress(ip > crit)) for ip in p] # 0 == nonsignificant

def groupcomp(data, groupby = None, control = None, mode = 'circular', include = {}, exclude = {}, kwarg = {}):
    
    stats_kw = dict(stats_cen = 'independent t test', test_var = True, stats_var = 'f test', 
                    crit_cen = [0.05,0.01,0.001,0.0001], crit_var = 0.05)
    
    for k, v in kwarg.items():
        stats_kw[k] = v

    # exclude group column, splice groups
    if groupby:
        data = data.set_index(groupby)
    if include.get('feature', {}):
        # If specific features are included
        features = [feature for feature in data.columns if feature in include['feature']]
    else:
        if exclude.get('feature',{}):
            # Exclude specific features
            features = [feature for feature in data.columns if feature not in exclude['feature']]
        else:
            features = data.columns
    # Same logic for group
    if include.get('group',{}):
        groups = [group for group in data.index.unique() if group in include['group']]
    else:
        if exclude.get('group',{}):
            groups = [group for group in data.index.unique() if group not in exclude['group']]
        else:
            groups = data.index.unique()
    ngroup = len(groups)
    nfeature = len(features)
    # stats
    error = np.ones((nfeature,ngroup))
    mean = np.ones((nfeature,ngroup))
    # Try vectorizing ?
    if stats_kw['stats_cen'] in ['median test', 'mannwhitneyu', 'wilcoxon']:
        # Use median for nonparametric test
        for igroup, group in enumerate(groups):
            error[:, igroup] = data.loc[group, features].std()
            mean[:, igroup] = data.loc[group, features].median()
    else:
        for igroup, group in enumerate(groups):
            error[:, igroup] = data.loc[group, features].std()
            mean[:, igroup] = data.loc[group, features].mean()
    mean = pd.DataFrame(mean,columns=groups,index=features)
    error = pd.DataFrame(error,columns=groups,index=features)
    if control:
        div = mean[control].copy()
        # Direct divide ?
        for group in groups:
            mean[group] = mean[group]/div
            error[group] = error[group]/div
    if mode == 'circular':
        level = np.ones((nfeature,ngroup))
        for igroup,group in enumerate(groups):
            nextgroup = (igroup+1)%ngroup
            if stats_kw['test_var']:
                equal_var = _test_var(stats_kw['stats_var'],stats_kw['crit_var'],data.loc[group,features],data.loc[groups[nextgroup],features])
            else:
                equal_var = [True for i in range(nfeature)]
            level[:,igroup] = _test_cen(stats_kw['stats_cen'],stats_kw['crit_cen'],equal_var,data.loc[group,features],data.loc[groups[nextgroup],features])
    return mean,error,level

def starplot(data, groupby = None, control = None, **kwargs):
    include = kwargs.get('include',{})
    exclude = kwargs.get('exclude',{})
    stats_kw = kwargs.get('stats_kw',{})
    mode = kwargs.get('mode','circular')
    plottype = kwargs.get('plottype','barplot')
    errorbar = kwargs.get('errorbar',True) 
    star = kwargs.get('star',['*','**','***','****'])
    plot_kw = kwargs.get('plot_kw', {})
    rotate = plot_kw.get('rotate',0)
    linewidth = plot_kw.get('linewidth',1)
    elinewidth = plot_kw.get('elinewidth',0.5)
    fontsize = plot_kw.get('fontsize',14)
    capsize = plot_kw.get('capsize',4)
    starsize = plot_kw.get('starsize',3)
    noffset_ylim = plot_kw.get('noffset_ylim',35)
    noffset_fst = plot_kw.get('noffset_fst',10)
    noffset_snd = plot_kw.get('noffset_snd',10)
    
    plt.rcParams['font.family'] = 'Times New Roman'
    fig,ax = plt.subplots()
    if plottype == 'barplot':
        mean,error,level = groupcomp(data, groupby, control , mode, include, exclude, stats_kw)
        if errorbar:
            plot = mean.plot.bar(yerr = error,ax = ax,rot=rotate,capsize = capsize, 
                                error_kw=dict(elinewidth = elinewidth),fontsize = fontsize)
            max_bar = (mean+error).values
            min_bar = (mean-error).values
        else:
            plot = mean.plot.bar(ax = ax,rot=rotate,capsize = capsize, 
                                error_kw=dict(elinewidth = elinewidth),fontsize = fontsize)
            max_bar = mean.values
            min_bar = mean.values
    offset = max(max_bar.flatten())/100
    ylim = 0
    blank = []
    nfeature,ngroup = level.shape
    if mode == 'circular':
        for igroup in range(ngroup-1):
            for ifeature in range(nfeature):
                height = 0
                if level[ifeature,igroup] > 0:
                    center = [plot.patches[igroup*nfeature+ifeature].get_x(), 
                            plot.patches[igroup*nfeature+nfeature+ifeature].get_x()]
                    h1 = max_bar[ifeature][igroup]
                    h2 = max_bar[ifeature][igroup+1]
                    height = max([h1,h2])
                    width = plot.patches[igroup*nfeature+ifeature].get_width()
                    x = [c+width/2 for c in center]
                    ymin = [h1+offset*2,h2+offset*2]
                    ymax = [height+noffset_fst*offset+(-1)**igroup*2*offset,
                            height+(noffset_fst+1)*offset+(-1)**igroup*2*offset]
                    blank.append((x[0],ymax[0]))
                    blank.append((x[1],ymax[0]))
                    ax.vlines(x = x[0],ymin = ymin[0],ymax = ymax[0],lw = linewidth)
                    ax.vlines(x = x[1],ymin = ymin[1],ymax = ymax[0],lw = linewidth)
                    ax.annotate(star[int(level[ifeature,igroup]-1)],xy = ((x[0]+x[1])/2,ymax[1]),ha='center',size = starsize)
        igroup = ngroup-1
        if igroup != 1:
            for ifeature in range(nfeature):
                if level[ifeature,igroup] != 0:
                    center = [plot.patches[ifeature].get_x(), 
                            plot.patches[igroup*nfeature+ifeature].get_x()]
                    h1 = max_bar[ifeature][0]
                    h2 = max_bar[ifeature][igroup]
                    height = max(max_bar[ifeature])
                    width = plot.patches[igroup*nfeature+ifeature].get_width()
                    x = [c+width/2 for c in center]
                    ymin = [h1+offset*2,h2+offset*2]
                    ymax = [height+(noffset_fst+noffset_snd)*offset+(-1)**igroup*2*offset,
                            height+(noffset_fst+noffset_snd+1)*offset+(-1)**igroup*2*offset]
                    blank.append((x[0],ymax[0]))
                    blank.append((x[1],ymax[0]))
                    ax.vlines(x = x[0],ymin = ymin[0],ymax = ymax[0],lw = linewidth)
                    ax.vlines(x = x[1],ymin = ymin[1],ymax = ymax[0],lw = linewidth)
                    ax.annotate(star[int(level[ifeature,igroup]-1)],xy = ((x[0]+x[1])/2,ymax[1]),ha='center',size = starsize)
            if height > ylim:
                ylim = height
    ax.set_ylim(min(0,min(min_bar.flatten())-10*offset),ylim+noffset_ylim*offset)
    ax.set_ylim()
    for ind,loc in enumerate(blank):
        ax.vlines(x = loc[0], ymin = loc[1],ymax = loc[1]+offset*2,color = 'white',lw = 1.2*linewidth)
        if ind%2 == 1:
            ax.hlines(y = loc[1], xmin = blank[ind-1], xmax = blank[ind],lw = linewidth)


"""
# splice groups
    groups = [None]
    index = []
    start = -1
    for ind,group in enumerate(data.loc[:,groupby]):
        if group != groups[-1]:
            groups.append(group)
            index.append(range(start,ind-1))
            start = ind
        elif ind == len(data.loc[:,groupby])-1:
            index.append(range(start,ind))        
    index.pop(0)
    groups.pop(0)
    rm = []
    for ind,group in enumerate(groups):
        if group in exclude:
            rm.append(ind)
    for diff,it in enumerate(rm):
        groups.pop(it-diff)
        index.pop(it-diff)

    ngroup = len(index)
def starplot(df = [],x = '',y = '',data = [],feature = [],group = [],
             fold = False,foldcol = 0,mode = 3, errorbar = True, 
             plottype = 'barplot', stats_cen = 'independent t test',
             test_var = False, stats_var = 'f test', crit_var = 0.05, equal_var = True,
             rotate = 0, elinewidth = 0.5, fontsize = 14, capsize = 4,
             noffset_ylim = 35, noffset_fst = 10,noffset_diff = 10,star_size = 3,linewidth = 1,
             crit = [0.05,0.01,0.001,0.0001],star = ['*','**','***','****']):
    # data: list of pd.DataFrames for comparison (row: obs, columns: feature)
    # index: var, columns: obs
    # adjacent: annotate star for adjacent bar
    # control: annotate star between all other bars to selctive control bar
    # mix: mix mode
    # 3: annotate star for all combination of bar (only 3 bars available)

    crit = np.array(crit)
    plt.rcParams['font.family'] = 'Times New Roman'
    fig,ax = plt.subplots()
    ngroup = len(data)
    nfeature = data[0].shape[1]
    if plottype == 'barplot':
        error = np.ones((nfeature,ngroup))
        mean = np.ones((nfeature,ngroup))
        for igroup in range(ngroup):
            error[:,igroup] = data[igroup].std()
            mean[:,igroup] = data[igroup].mean()
        error = pd.DaraFrame(error)
        mean = pd.DataFrame(mean)
        if len(feature) != 0:
            error.index = feature
            mean.index = feature
        if len(group) != 0:
            error.columns = group
            mean.columns = group
        if fold:
            div = mean.iloc[:,foldcol]
            for ifeature in range(feature):
                mean.iloc[:,ifeature] = mean.iloc[:,ifeature]/div
                error.iloc[:,ifeature] = error.iloc[:,ifeature]/div
        if errorbar:
            plot = mean.plot.bar(yerr = error,ax = ax,rot=rotate,capsize = capsize, 
                                error_kw=dict(elinewidth = elinewidth),fontsize = fontsize)
            max_bar = mean+error
            min_bar = mean-error
        else:
            plot = mean.plot.bar(ax = ax,rot=rotate,capsize = capsize, 
                                error_kw=dict(elinewidth = elinewidth),fontsize = fontsize)
            max_bar = mean
            min_bar = mean
    elif plottype == 'boxplot':
        print("under buiding")
    ylim = 0
    offset = max(max_bar)/100
    blank = []
    if mode == 3:
        for igroup in range(ngroup):
            nextgroup = (igroup+1)%3
            if test_var:
                equal_var = _test_var(stats_var,crit_var,data[igroup],data[nextgroup])
            level = _test_cen(stats,crit,equal_var,data[igroup],data[nextgroup])
            for ifeature in range(nfeature):
                height = 0
                if level[igroup] != 0 and igroup != ngroup-1:
                    center = [plot.patches[igroup*nfeature+ifeature].get_x(), plot.patches[igroup*nfeature+nfeature+ifeature].get_x()]
                    height = max([max_bar[ifeature][igroup],max_bar[ifeature][igroup+1]])
                    h1 = max_bar[ifeature][igroup]
                    h2 = max_bar[ifeature][igroup+1]
                    width = plot.patches[igroup*nfeature+ifeature].get_width()
                    blank.append((center[0]+width/2,height+noffset_fst*offset+(-1)**igroup*2*offset))
                    blank.append((center[1]+width/2,height+noffset_fst*offset+(-1)**igroup*2*offset))
                    ax.vlines(x = center[0]+width/2,ymin = h1+offset*2,ymax = height+noffset_fst*offset+(-1)**igroup*2*offset,lw = linewidth)
                    ax.vlines(x = center[1]+width/2,ymin = h2+offset*2,ymax = height+noffset_fst*offset+(-1)**igroup*2*offset,lw = linewidth)
                    ax.annotate(star[int(level[igroup]-1)],
                                xy = ((center[0]+center[1])/2+width/2,height+(noffset_fst+1)*offset+(-1)**igroup*2*offset),
                                ha='center',size = star_size)
                elif level[igroup] != 0 and igroup == ngroup-1:
                    center = [plot.patches[ifeature].get_x(), plot.patches[igroup*m+ifeature].get_x()]
                    height = max(max_bar[ifeature])
                    h1 = max_bar[ifeature][0]
                    h2 = max_bar[ifeature][igroup]
                    blank.append((center[0]+width/2,height+(noffset_fst+noffset_diff)*offset))
                    blank.append((center[1]+width/2,height+20*offset))
                    ax.vlines(x = center[0]+width/2,ymin = h1+offset*2,ymax = height+(noffset_fst+noffset_diff)*offset,lw = linewidth)
                    ax.vlines(x = center[1]+width/2,ymin = h2+offset*2,ymax = height+(noffset_fst+noffset_diff)*offset,lw = linewidth)
                    ax.annotate(star[int(level[igroup]-1)],
                                xy = ((center[0]+center[1])/2+width/2,height+(noffset_fst+noffset_diff+1)*offset),
                                ha='center',size = star_size)
                if height > ylim:
                    ylim = height
    if mode == 'adjacent':
        for ifeature in range(nfeature):
            level = np.zeros(ngroup-1)
            for igroup in range(ngroup-1):
                nextgroup = igroup+1
                if test_var == True:
                    if stats_var == 'f test':
                        f = 0.5-abs(0.5-ftest.sf(data[igroup][:,ifeature].var()/data[nextgroup][:,ifeature].var(),len(data[igroup][:,ifeature])-1,len(data[nextgroup][:,ifeature])-1))
                        if crit_var/2 > f:
                            equal_var = False
                        else:
                            equal_var = True
                    else:
                        if stats_var == 'bartlett':
                            f = bartlett(data[igroup][:,ifeature],data[nextgroup][:,ifeature])[1]
                        elif stats_var == 'levene':
                            f = bartlett(data[igroup][:,ifeature],data[nextgroup][:,ifeature])[1]
                        elif stats_var == 'fligner':
                            f = fligner(data[igroup][:,ifeature],data[nextgroup][:,ifeature])[1]
                        if crit_var > f:
                            equal_var = False
                        else:
                            equal_var = True
                if stats == 'independent t test':
                    p = ttest_ind(data[igroup][:,ifeature],data[nextgroup][:,ifeature],equal_var = equal_var)[1]
                elif stats == 'paired t test':
                    if equal_var == True:
                        p = ttest_rel(data[igroup][:,ifeature],data[nextgroup][:,ifeature])[1]
                    else:
                        p = 0
                elif stats == 'median test':
                    p = median_test(data[igroup][:,ifeature],data[nextgroup][:,ifeature])[1]
                elif stats == 'mannwhitneyu':
                    if equal_var == True:
                        p = mannwhitneyu(data[igroup][:,ifeature],data[nextgroup][:,ifeature])[1]
                    else:
                        p = 0
                elif stats == 'wilcoxon':
                    if equal_var == True:
                        p = wilcoxon(data[igroup][:,ifeature],data[nextgroup][:,ifeature])[1]
                    else:
                        p = 0
                level[igroup] = len(crit) - len(crit.compress(p > crit))
            for igroup in range(ngroup-1):
                height = 0
                if level[igroup] != 0:
                    center = [plot.patches[igroup*nfeature+ifeature].get_x(), plot.patches[igroup*nfeature+nfeature+ifeature].get_x()]
                    height = max([max_bar[ifeature][igroup],max_bar[ifeature][igroup+1]])
                    h1 = max_bar[ifeature][igroup]
                    h2 = max_bar[ifeature][igroup+1]
                    width = plot.patches[igroup*nfeature+ifeature].get_width()
                    blank.append((center[0]+width/2,height+noffset_fst*offset+(-1)**igroup*2*offset))
                    blank.append((center[1]+width/2,height+noffset_fst*offset+(-1)**igroup*2*offset))
                    ax.vlines(x = center[0]+width/2,ymin = h1+offset*2,ymax = height+noffset_fst*offset+(-1)**igroup*2*offset,lw = linewidth)
                    ax.vlines(x = center[1]+width/2,ymin = h2+offset*2,ymax = height+noffset_fst*offset+(-1)**igroup*2*offset,lw = linewidth)
                    ax.annotate(star[int(level[igroup]-1)],
                                xy = ((center[0]+center[1])/2+width/2,height+(noffset_fst+1)*offset+(-1)**igroup*2*offset),
                                ha='center',size = star_size)
                if height > ylim:
                    ylim = height
    ax.set_ylim(min(0,min(min_bar)-10*offset),ylim+noffset_ylim*offset)
    for j,i in enumerate(blank):
        ax.vlines(x = i[0], ymin = i[1],ymax = i[1]+offset*2,color = 'white',lw = 1.2*linewidth)
        if j%2 == 1:
            ax.hlines(y = i[1], xmin = blank[j-1], xmax = blank[j],lw = linewidth)
"""