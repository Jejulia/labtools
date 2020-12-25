# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import tkinter as tk
class package:
    def __init__(self):
        #   elements defined
        C = 12
        H = 1.007825
        N = 14.003074
        O = 15.994915
        P = 30.973763
        S = 31.972072
        Na = 22.98977
        Cl = 34.968853
        self.elements = [C,H,N,O,P,S,Na,Cl]
        self.elementsymbol = ['C','H','N','O','P','S','Na','Cl']
        ionname = ['M','M+H','M+2H','M+H-H2O','M+2H-H2O','M+Na','M+2Na','M+2Na-H','M+NH4',
                   'M-H','M-2H','M-3H','M-4H','M-5H','M-H-H2O','M-2H-H2O','M-CH3','M+Cl','M+HCOO','M+OAc']
        ionfunc = []
        ionfunc.append(lambda ms: ms)
        ionfunc.append(lambda ms: ms+package().elements[1])
        ionfunc.append(lambda ms: (ms+2*package().elements[1])/2)
        ionfunc.append(lambda ms: ms-package().elements[1]-package().elements[3])
        ionfunc.append(lambda ms: (ms-package().elements[3])/2)
        ionfunc.append(lambda ms: ms+package().elements[6])
        ionfunc.append(lambda ms: (ms+2*package().elements[6])/2)
        ionfunc.append(lambda ms: ms-package().elements[1]+2*package().elements[6])
        ionfunc.append(lambda ms: ms+4*package().elements[1]+package().elements[2])
        ionfunc.append(lambda ms: ms-package().elements[1])
        ionfunc.append(lambda ms: (ms-2*package().elements[1])/2)
        ionfunc.append(lambda ms: (ms-3*package().elements[1])/3)
        ionfunc.append(lambda ms: (ms-4*package().elements[1])/4)
        ionfunc.append(lambda ms: (ms-5*package().elements[1])/5)
        ionfunc.append(lambda ms: ms-3*package().elements[1]-package().elements[3])
        ionfunc.append(lambda ms: (ms-4*package().elements[1]-package().elements[3])/2)
        ionfunc.append(lambda ms: ms-package().elements[0]-3*package().elements[1])
        ionfunc.append(lambda ms: ms+package().elements[7])
        ionfunc.append(lambda ms: ms+package().elements[0]+package().elements[1]+2*package().elements[3])
        ionfunc.append(lambda ms: ms+2*package().elements[0]+3*package().elements[1]+2*package().elements[3])
        self.ion = {}
        for i,j in enumerate(ionname):
            self.ion[j] = ionfunc[i]
# %% [markdown]
# Package for Sphingolipids 

# %%

class package_sl(package):
    def __init__(self):
        #   base structure defined 
        self.base = {'Cer': np.array([0,3,1,0]+[0]*(len(package().elements)-4)),
                     'Sphingosine': np.array([0,3,1,0]+[0]*(len(package().elements)-4)),
                     'Sphinganine': np.array([0,3,1,0]+[0]*(len(package().elements)-4))}
        #   headgroups defined
        headgroup = ['Pi','Choline','Ethanolamine','Inositol','Glc','Gal','GalNAc','NeuAc','Fuc','NeuGc']
        formula = []
        formula.append(np.array([0,3,0,4,1]+[0]*(len(package().elements)-5)))
        formula.append(np.array([5,13,1,1]+[0]*(len(package().elements)-4)))
        formula.append(np.array([2,7,1,1]+[0]*(len(package().elements)-4)))
        formula.append(np.array([6,12,0,6]+[0]*(len(package().elements)-4)))
        formula.append(np.array([6,12,0,6]+[0]*(len(package().elements)-4)))
        formula.append(np.array([6,12,0,6]+[0]*(len(package().elements)-4)))
        formula.append(np.array([8,15,1,6]+[0]*(len(package().elements)-4)))
        formula.append(np.array([11,19,1,9]+[0]*(len(package().elements)-4)))
        formula.append(np.array([6,12,0,5]+[0]*(len(package().elements)-4)))
        formula.append(np.array([11,19,1,10]+[0]*(len(package().elements)-4)))
        self.components = self.base.copy()
        for i,j in enumerate(headgroup):
            self.components[j] = formula[i]
        #   sn type defined
        sntype = ['none','d','t']
        snformula = []
        snformula.append(lambda carbon,db: np.array([carbon,2*carbon-2*db,0,2]+[0]*(len(package().elements)-4)))
        snformula.append(lambda carbon,db: np.array([carbon,2*carbon+2-2*db,0,3]+[0]*(len(package().elements)-4)))
        snformula.append(lambda carbon,db: np.array([carbon,2*carbon+2-2*db,0,4]+[0]*(len(package().elements)-4)))
        self.sn = {}
        for i,j in enumerate(sntype):
            self.sn[j] = snformula[i]
        #   extended structure
        nana = ['M','D','T','Q','P']
        iso = ['1a','1b','1c']
        namedf = pd.DataFrame({'0-series': ['LacCer'],'a-series': ['GM3'],'b-series': ['GD3'],'c-series': ['GT3']})
        namedf = namedf.append(pd.Series(['G'+'A'+'2' for name in namedf.iloc[0,0:1]]+['G'+i+'2' for i in nana[0:3]],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series(['G'+'A'+'1' for name in namedf.iloc[0,0:1]]+['G'+i+j for i,j in zip(nana[0:3],iso)],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series(['G'+'M'+'1b' for name in namedf.iloc[0,0:1]]+['G'+i+j for i,j in zip(nana[1:4],iso)],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series(['G'+'D'+'1c' for name in namedf.iloc[0,0:1]]+['G'+i+j for i,j in zip(nana[2:],iso)],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series(['G'+'D'+'1α' for name in namedf.iloc[0,0:1]]+[i+'α' for i in namedf.iloc[4,1:]],index = namedf.columns), ignore_index=True)
        sequencedf = pd.DataFrame({'0-series': ['Gal-Glc-Cer'],'a-series': ['(NeuAc)-Gal-Glc-Cer'],'b-series': ['(NeuAc-NeuAc)-Gal-Glc-Cer'],'c-series': ['(NeuAc-NeuAc-NeuAc)-Gal-Glc-Cer']})
        sequencedf = sequencedf.append(pd.Series(['GalNAc-'+formula for formula in sequencedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['Gal-'+formula for formula in sequencedf.iloc[1,:]],index = namedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['NeuAc-'+formula for formula in sequencedf.iloc[2,:]],index = namedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['NeuAc-'+formula for formula in sequencedf.iloc[3,:]],index = namedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['NeuAc-Gal-(NeuAc)-GalNAc-'+formula for formula in sequencedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        self.base = {'Cer': 'Cer','Sphingosine': 'Sphingosine','Sphinganine': 'Sphinganine','Sphingosine-1-Phosphate': 'Pi-Sphingosine','Sphinganine-1-Phosphate': 'Pi-Sphinganine',
                     'CerP': 'Pi-Cer','SM': 'Choline-Pi-Cer','CerPEtn': 'Ethanolamine-Pi-Cer','CerPIns': 'Inositol-Pi-Cer',
                     'LysoSM(dH)': 'Choline-Pi-Sphinganine','LysoSM': 'Choline-Pi-Sphingosine',
                     'GlcCer': 'Glc-Cer','GalCer': 'Gal-Cer'}
        for i in namedf:
            for j,k in enumerate(namedf[i]):
                self.base[k] = sequencedf[i][j]
    def basesn(self,base,typ):
        typ = base[typ].split('-')[-1]
        if 'Cer' == base[typ]:
            return [['d','t'],list(range(18,23)),':',[0,1],'/',['none','h'],list(range(12,33)),':',[0,1]]
        elif 'Sphingosine' == base[typ]:
            return [['d','t'],list(range(18,23)),':','1']
        elif 'Sphinganine' == base[typ]:
            return [['d','t'],list(range(18,23)),':','0']
        else:
            return 0
    def iterate(self,base,typ,start,end):
        typ = base[typ].split('-')[-1]
        start = pd.Series(start)
        end = pd.Series(end)
        start = start.replace('none','')
        end = end.replace('none','')
        if 'Cer' == base[typ]:
            return ['{}{}:{}/{}{}:{}'.format(i,j,k,l,m,n) for i in [start[0]] for k in range(int(start[2]),int(end[2])+1) for j in range(int(start[1]),int(end[1])+1) for n in range(int(start[5]),int(end[5])+1) for l in [start[3]] for m in range(int(start[4]),int(end[4])+1)]
        elif 'Sphingosine' == base[typ]:
            return ['{}{}:1'.format(i,j) for i in [start[0]] for j in range(int(start[1]),int(end[1])+1)]
        elif 'Sphinganine' == base[typ]:
            return ['{}{}:0'.format(i,j) for i in [start[0]] for j in range(int(start[1]),int(end[1])+1)]
        else:
            return 0
# %% [markdown]
# Package for Glycerophospholipids

# %%
class package_gpl(package):
    def __init__(self):
        #   base structure defined 
        self.base = {'PA':  np.array([3,9,0,6,1]+[0]*(len(package().elements)-5)),
                     'LysoPA': np.array([3,9,0,6,1]+[0]*(len(package().elements)-5))}
        #   headgroups defined
        headgroup = ['Pi','Choline','Ethanolamine','Inositol','Glycerol']
        formula = []
        formula.append(np.array([0,3,0,4,1]+[0]*(len(package().elements)-5)))
        formula.append(np.array([5,13,1,1]+[0]*(len(package().elements)-4)))
        formula.append(np.array([2,7,1,1]+[0]*(len(package().elements)-4)))
        formula.append(np.array([6,12,0,6]+[0]*(len(package().elements)-4)))
        formula.append(np.array([3,8,0,3]+[0]*(len(package().elements)-4)))
        self.components = self.base.copy()
        for i,j in enumerate(headgroup):
            self.components[j] = formula[i]
        #   sn type defined
        sntype = ['none','O','P']
        snformula = []
        snformula.append(lambda carbon,db: np.array([carbon,2*carbon-2*db,0,2]+[0]*(len(package().elements)-4)))
        snformula.append(lambda carbon,db: np.array([carbon,2*carbon+2-2*db,0,1]+[0]*(len(package().elements)-4)))
        snformula.append(lambda carbon,db: np.array([carbon,2*carbon-2*db,0,1]+[0]*(len(package().elements)-4)))
        self.sn = {}
        for i,j in enumerate(sntype):
            self.sn[j] = snformula[i]
        #   extended structure(extended structure can be defined by library.baseext())
        namedf = pd.DataFrame({'a': ['PA'],'b': ['LysoPA']})
        namedf = namedf.append(pd.Series([name[0:-1]+'C' for name in namedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series([name[0:-1]+'E' for name in namedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series([name[0:-1]+'G' for name in namedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series([name[0:-1]+'GP' for name in namedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series([name[0:-1]+'I' for name in namedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series([name[0:-1]+'IP' for name in namedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series([name[0:-1]+'IP2' for name in namedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        namedf = namedf.append(pd.Series([name[0:-1]+'IP3' for name in namedf.iloc[0,:]],index = namedf.columns), ignore_index=True)
        sequencedf = pd.DataFrame({'a': ['PA'],'b': ['LysoPA']})
        sequencedf = sequencedf.append(pd.Series(['Choline-'+name for name in sequencedf.iloc[0,:]],index = sequencedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['Ethanolamine-'+name for name in sequencedf.iloc[0,:]],index = sequencedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['Glycerol-'+name for name in sequencedf.iloc[0,:]],index = sequencedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['Pi-'+name for name in sequencedf.iloc[3,:]],index = sequencedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['Inositol-'+name for name in sequencedf.iloc[0,:]],index = sequencedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['Pi-'+name for name in sequencedf.iloc[5,:]],index = sequencedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['Pi-'+name for name in sequencedf.iloc[6,:]],index = sequencedf.columns), ignore_index=True)
        sequencedf = sequencedf.append(pd.Series(['Pi-'+name for name in sequencedf.iloc[7,:]],index = sequencedf.columns), ignore_index=True)
        self.base = {'PA': 'PA','LysoPA': 'LysoPA'}
        for i in namedf:
            for j,k in enumerate(namedf[i]):
                self.base[k] = sequencedf[i][j]
    def basesn(self,base,typ):
        typ = base[typ].split('-')[-1]
        if 'PA' == base[typ]:
            return [['none','O','P'],list(range(2,27)),':',[0,1,2,3,4,5,6],'/',['none','O','P'],list(range(2,27)),':',[0,1,2,3,4,5,6]]
        elif 'LysoPA' == base[typ]:
            return [['none','O','P'],list(range(2,27)),':',[0,1,2,3,4,5,6]]
        else:
            return 0
    def iterate(self,base,typ,start,end):
        typ = base[typ].split('-')[-1]
        start = pd.Series(start)
        end = pd.Series(end)
        start = start.replace('none','')
        end = end.replace('none','')
        if 'PA' == base[typ]:
            return ['{}{}:{}/{}{}:{}'.format(i,j,k,l,m,n) for i in [start[0]] for j in range(int(start[1]),int(end[1])+1) for k in range(int(start[2]),int(end[2])+1) for l in [start[3]]  for m in range(int(start[4]),int(end[4])+1) for n in range(int(start[5]),int(end[5])+1)]
        elif 'LysoPA' == base[typ]:
            return ['{}{}:{}'.format(i,j,k) for i in [start[0]] for j in range(int(start[1]),int(end[1])+1) for k in range(int(start[2]),int(end[2])+1)]
        else:
            return 0
# %% [markdown]
# library class

# %%
class library(package):
    def __init__(self,pack):
        self.elements = package().elements
        self.elementsymbol = package().elementsymbol
        self.ion = package().ion
        self.components = {}
        self.base = {}
        self.sn = {}
        self.basesnorg = []
        self.iterateorg = []
        for i,j in enumerate(pack):
            self.components = {**self.components,**j().components}
            self.base = {**self.base,**j().base}
            self.sn = {**self.sn,**j().sn}
            self.basesnorg.append(j().basesn)
            self.iterateorg.append(j().iterate)
    def basesn(self,typ):
        base = self.base
        for i in range(len(self.basesnorg)):
            if not self.basesnorg[i](base,typ) == 0:
                return self.basesnorg[i](base,typ)
    def iterate(self,typ,start,end):
        base = self.base
        for i in range(len(self.iterateorg)):
            if not self.iterateorg[i](base,typ,start,end) == 0:
                return self.iterateorg[i](base,typ,start,end)
    def newhgdef(self,newheadgroup,newformula):
        self.components[newheadgroup] = newformula
    def baseext(self,name,sequence):
        self.base[name] = sequence
    def mscomp(self,name):
        components = name.split('-')
        base = components[-1].split('(')[0]
        sn = components[-1].split('(')[1].split(')')[0].split('/')
        hg = '('+name.replace(base,'')+self.base[base]+')'
        hgcode = []
        s = 0
        hg = hg.split('-')
        hg.reverse()
        for i,j in enumerate(hg):
            if ')' in j:
                s += 1
            hgcode.append(s)
            if '(' in j:
                s+= -1
            hg[i] = j.replace('(','').replace(')','')
        code = []
        for i,j in enumerate(hgcode):
            if i == 0:
                code.append([0])
            elif hgcode[i-1] == j:
                new = code[i-1].copy()
                last = new[-1]+1
                new.pop()
                new.append(last)
                code.append(new)
            elif hgcode[i-1] < j:
                new = code[i-1].copy()
                new.append(0)
                code.append(new)
            elif hgcode[i-1] > j:
                pre = max([k for k in range(i) if hgcode[k] == j])
                new = code[pre].copy()
                last = new[-1]+1
                new.pop()
                new.append(last)
                code.append(new)
        comp = pd.DataFrame({'headgroups': hg,'position': code})
        return comp
    def msformula(self,name,mode):
        components = name.split('-')
        base = components[-1].split('(')[0]
        sn = components[-1].split('(')[1].split(')')[0].split('/')
        headgroups = components[0:-1]
        for hg in headgroups:
            if '(' in hg:
                if ')' not in hg.split('(')[1]:
                    headgroups[headgroups.index(hg)] = hg.split('(')[1]
                elif ')' in hg.split('(')[1]:
                    headgroups[headgroups.index(hg)] = hg.split('(')[1].split(')')[0]
            elif ')' in hg:
                headgroups[headgroups.index(hg)] = hg.split(')')[0]
        ms = np.array([0,2,0,1]+[0]*(len(self.elements)-4))
        H2O = np.array([0,2,0,1]+[0]*(len(self.elements)-4))
        for hg in headgroups:
            ms += self.components[hg]
            ms += -H2O 
        components = self.base[base].split('-')
        for c in components:
            if '(' in c:
                if ')' not in c.split('(')[1]:
                    components[components.index(c)] = c.split('(')[1]
                elif ')' in c.split('(')[1]:
                    components[components.index(c)] = c.split('(')[1].split(')')[0]
            elif ')' in c:
                components[components.index(c)] = c.split(')')[0]
        for c in components:
            ms += self.components[c]
            ms += -H2O 
        for sni in sn:
            if 'd' in sni:
                carbon = int(sni.split('d')[1].split(':')[0])
                db = int(sni.split('d')[1].split(':')[1])
                ms += self.sn['d'](carbon,db)
            elif 't' in sni:
                carbon = int(sni.split('t')[1].split(':')[0])
                db = int(sni.split('t')[1].split(':')[1])
                ms += self.sn['t'](carbon,db)
            elif 'O' in sni:
                carbon = int(sni.split('O')[1].split(':')[0])
                db = int(sni.split('O')[1].split(':')[1])
                ms += self.sn['O'](carbon,db)
            elif 'P' in sni:
                carbon = int(sni.split('P')[1].split(':')[0])
                db = int(sni.split('P')[1].split(':')[1])
                ms += self.sn['P'](carbon,db)
            else:
                carbon = int(sni.split(':')[0])
                db = int(sni.split(':')[1])
                ms += self.sn['none'](carbon,db)
            ms += -H2O
        if mode == 'raw':
            return ms
        elif mode == 'molecule':
            formulalist = [i+'{}'.format(j) for i,j in zip(self.elementsymbol[0:len(ms)],ms) if j > 0]
            formula = ''
            for f in formulalist:
                formula += f
            return formula
    def mscalculator(self,name,ion):
        ms = (self.msformula(name,mode='raw')*self.elements[0:len(self.msformula(name,mode='raw'))]).cumsum()[-1]
        return self.ion[ion](ms)
    def export(self):
        expwind = tk.Tk()
        expwind.title('Export settings')
        expwind.geometry('700x300')
        var_base = tk.StringVar()
        initialbase = list(self.base.keys())
        title = tk.Label(expwind,text = 'Select base')
        title.config(font=("Times New Roman", 20))
        var_base.set(initialbase)
        listbox1 = tk.Listbox(expwind,listvariable = var_base,selectmode = 'extended')
        listbox1.config(font=("Times New Roman", 12))
        var_add = tk.StringVar()
        subtitle = tk.Label(expwind,text = 'others')
        subtitle.config(font=("Times New Roman", 15))
        other = tk.Entry(expwind,textvariable = var_add)
        other.config(font=("Times New Roman", 12))
        def base_selection():
            global base_input 
            base_input = [listbox1.get(i) for i in listbox1.curselection()]
            title.destroy()
            listbox1.destroy()
            button1.destroy() 
            subtitle.destroy()
            other.destroy()
            addbutton.destroy()
            global sn_input
            sn_input = []
            def snloop(i,skip,add,apply):
                if skip == True:
                    i += 1
                else:
                    global menu_st,menu_end,var_st,var_end
                    menu_st = []
                    menu_end = []
                    var_st = []
                    var_end = []
                    title = tk.Label(expwind,text = base_input[i])
                    title.config(font=("Times New Roman", 20))
                    title.grid(row = 0,column = 0,padx=20)
                    labelstart = tk.Label(expwind,text = 'start')
                    labelstart.config(font=("Times New Roman", 15))
                    labelend = tk.Label(expwind,text = 'end')
                    labelend.config(font=("Times New Roman", 15))
                    labelstart.grid(row = 1,column = 0,padx=20)
                    label = []
                    for n,sntype in enumerate(self.basesn(base_input[i])):
                        if type(sntype) == str:
                            label.append(tk.Label(expwind,text = sntype))
                            label[-1].config(font=("Times New Roman", 12))
                            label[-1].grid(row = 1,column = n+1)
                        else:
                            var_st.append(tk.StringVar())
                            menu_st.append(tk.OptionMenu(expwind,var_st[-1],*sntype))
                            menu_st[-1].config(font=("Times New Roman", 12))
                            menu_st[-1].grid(row = 1,column = n+1)
                    labelend.grid(row = 2,column = 0,padx=20)
                    for n,sntype in enumerate(self.basesn(base_input[i])):
                        if type(sntype) == str:
                            label.append(tk.Label(expwind,text = sntype))
                            label[-1].config(font=("Times New Roman", 12))
                            label[-1].grid(row = 2,column = n+1)
                        elif type(sntype[0]) == str:
                            label.append(tk.Label(expwind,text = ''))
                            label[-1].config(font=("Times New Roman", 12))
                            label[-1].grid(row = 2,column = n+1)
                            var_end.append(tk.StringVar())
                            var_end[-1].set(var_st[n].get())
                            menu_end.append(tk.OptionMenu(expwind,var_end[-1],*sntype))
                        else:
                            var_end.append(tk.StringVar())
                            menu_end.append(tk.OptionMenu(expwind,var_end[-1],*sntype))
                            menu_end[-1].config(font=("Times New Roman", 12))
                            menu_end[-1].grid(row = 2,column = n+1)
                    i += 1
                def sn_selection():
                    st = []
                    end = []
                    for n in range(len(menu_st)):
                        st.append(var_st[n].get())
                        end.append(var_end[n].get())
                        menu_st[n].destroy()
                        menu_end[n].destroy()
                    for n in label:
                        n.destroy()
                    title.destroy()
                    labelstart.destroy()
                    labelend.destroy()
                    button2.destroy()
                    button3.destroy()
                    button4.destroy()
                    if add == True:
                        sn_input[-1] = sn_input[-1]+self.iterate(base_input[i-1],st,end)
                    else:
                        sn_input.append(self.iterate(base_input[i-1],st,end))
                    if i  < len(base_input):
                        snloop(i,skip = False,add = False,apply = False)
                    else:
                        cancel.destroy()
                        ion_selection()
                def apply_all():
                    st = []
                    end = []
                    for n in range(len(menu_st)):
                        st.append(var_st[n].get())
                        end.append(var_end[n].get())
                        menu_st[n].destroy()
                        menu_end[n].destroy()
                    for n in label:
                        n.destroy()
                    title.destroy()
                    labelstart.destroy()
                    labelend.destroy()
                    if apply == False:
                        button2.destroy()
                    button3.destroy()
                    button4.destroy()
                    if add == True:
                        sn_input[-1] = sn_input[-1]+self.iterate(base_input[i-1],st,end)
                    else:
                        sn_input.append(self.iterate(base_input[i-1],st,end))
                    if i  < len(base_input):
                        if self.basesn(base_input[i]) in [self.basesn(base_input[p]) for p in range(i)]:
                            snloop(i,skip = True,add = False,apply = True)
                        else: 
                            snloop(i,skip = False,add = False,apply = True)
                    else:
                        ion_selection()
                def add_other():
                    st = []
                    end = []
                    for n in range(len(menu_st)):
                        st.append(var_st[n].get())
                        end.append(var_end[n].get())
                        menu_st[n].destroy()
                        menu_end[n].destroy()
                    for n in label:
                        n.destroy()
                    title.destroy()
                    labelstart.destroy()
                    labelend.destroy()
                    if apply == False:
                        button2.destroy()
                    button3.destroy()
                    button4.destroy()
                    if add == True:
                        sn_input[-1] = sn_input[-1]+self.iterate(base_input[i-1],st,end)
                    else:
                        sn_input.append(self.iterate(base_input[i-1],st,end))
                    if apply == True:
                        snloop(i-1,skip = False,add = True,apply = True)
                    else:
                        snloop(i-1,skip = False,add = True,apply = False)
                if skip == False:
                    if apply == False:
                        button2 = tk.Button(expwind,text = 'confirm',command = sn_selection)
                        button2.config(font=("Times New Roman", 12))
                        button2.grid(row = 3,column = 0)
                    button3 = tk.Button(expwind,text = 'apply to others',command = apply_all)
                    button3.config(font=("Times New Roman", 12))
                    button3.grid(row = 3,column = 1)
                    button4 = tk.Button(expwind,text = 'add',command = add_other)
                    button4.config(font=("Times New Roman", 12))
                    button4.grid(row = 3,column = 2)
                    expwind.mainloop()
                else:
                    index = [self.basesn(base_input[p]) for p in range(i-1)].index(self.basesn(base_input[i-1]))
                    sn_input.append(sn_input[index])
                    if i  < len(base_input):
                        if self.basesn(base_input[i]) in [self.basesn(base_input[p]) for p in range(i)]:
                            snloop(i,skip = True,add = False,apply = False)
                        else: 
                           snloop(i,skip = False,add = False,apply = False)
                    else:
                        cancel.destroy()
                        ion_selection()
            snloop(0,skip = False,add = False,apply = False)
        def add_base():
            initialbase.append(other.get())
            var_base.set(initialbase)
            other.delete(first = 0,last = 100)
        def ion_selection():
            title1 = tk.Label(expwind,text = 'Select ions')
            title1.config(font=("Times New Roman", 20))
            var_ion = tk.StringVar()
            var_ion.set(list(package().ion.keys()))
            listbox2 = tk.Listbox(expwind,listvariable = var_ion,selectmode = 'extended')
            listbox2.config(font=("Times New Roman", 12))
            title1.grid(row = 0,column = 0,padx=100)
            listbox2.grid(row = 1,column = 0,padx=100)
            def filename_type():
                global ion_input
                ion_input = [listbox2.get(i) for i in listbox2.curselection()]
                title1.destroy()
                listbox2.destroy()
                button5.destroy()
                title2 = tk.Label(expwind,text = 'Type filename(with.xlsx)')
                title2.config(font=("Times New Roman", 20))
                var_file = tk.StringVar(value = 'library.xlsx')
                file = tk.Entry(expwind,textvariable = var_file)
                file.config(font=("Times New Roman", 12))
                title2.grid(row = 0,column = 0,padx=100)
                file.grid(row = 1,column = 0,padx=100)
                def export():
                    global filename
                    filename = var_file.get()
                    self.toexcel()
                    expwind.destroy()
                    root()
                button6 = tk.Button(expwind,text = 'export',command = export)
                button6.config(font=("Times New Roman", 12))
                button6.grid(row = 2,column = 0,padx=100)
                expwind.mainloop()
            button5 = tk.Button(expwind,text = 'confirm',command = filename_type)
            button5.config(font=("Times New Roman", 12))
            button5.grid(row = 2,column = 0,padx=100)
            cancel = tk.Button(expwind,text = 'cancel',command = cancelrun)
            cancel.config(font=("Times New Roman", 12))
            cancel.grid(row = 2,column = 1,padx=5)
            expwind.mainloop()
        def cancelrun():
            expwind.destroy()
            root()
        button1 = tk.Button(expwind,text = 'confirm',command = base_selection)
        button1.config(font=("Times New Roman", 12))
        addbutton = tk.Button(expwind,text = 'add',command = add_base)
        addbutton.config(font=("Times New Roman", 12))
        title.grid(row = 0,column = 0,padx=20)
        listbox1.grid(row = 1,column = 0,rowspan = 9,padx=20)
        button1.grid(row = 10,column = 0,padx=20)
        subtitle.grid(row = 0,column = 1,padx=20)
        other.grid(row = 1,column = 1,padx=20)
        addbutton.grid(row = 2,column = 1,padx=20)
        cancel = tk.Button(expwind,text = 'cancel',command = cancelrun)
        cancel.config(font=("Times New Roman", 12))
        cancel.grid(row = 10,column = 20)
        expwind.mainloop()
    def toexcel(self):
        with pd.ExcelWriter(filename) as writer:
            self.df = {}
            for i,b in enumerate(base_input):
                name = [b+'('+j+')' for j in sn_input[i]]
                self.df[b] = pd.DataFrame({b: name})
                self.df[b]['formula'] = [self.msformula(j,'molecule') for j in name]
                for ion in ion_input:
                    ms = [self.mscalculator(i,ion) for i in self.df[b][b]]
                    self.df[b][ion] = ms
                self.df[b].to_excel(writer,index = False,sheet_name = '{}'.format(b))
# %% [markdown]
# GUI

# %%
def entry():
    package_available = {'Sphingolipid': package_sl,'Glycerophospholipid': package_gpl}
    entrywind = tk.Tk()
    entrywind.geometry('500x300')
    title = tk.Label(entrywind,text = 'Welcome! choose package(s)')
    title.config(font=("Times New Roman", 20))
    var_pack = tk.StringVar()
    var_pack.set(list(package_available.keys()))
    listbox = tk.Listbox(entrywind,listvariable = var_pack,selectmode = 'extended')
    def choose_pack():
        pack = [listbox.get(i) for i in listbox.curselection()]
        global currentlibrary
        currentlibrary = library([package_available[i] for i in pack])
        entrywind.destroy()
        root()
    button = tk.Button(entrywind,text = 'confirm',command = choose_pack)
    button.config(font=("Times New Roman", 12))
    title.pack()
    listbox.pack()
    button.pack()
    entrywind.mainloop()
def root():
    rootwind = tk.Tk()
    rootwind.geometry('500x300')
    def run_export():
        rootwind.destroy()
        global exp
        exp = True
    def cancelrun():
        rootwind.destroy()
        global exp
        exp = False
        del currentlibrary
    title = tk.Label(rootwind,text = 'Root')
    title.config(font=("Times New Roman", 20))
    export = tk.Button(rootwind,text = 'Export data',command = run_export)
    export.config(font=("Times New Roman", 12))
    cancel = tk.Button(rootwind,text = 'cancel',command = cancelrun)
    cancel.config(font=("Times New Roman", 12))
    title.pack(pady = 5)
    export.pack(pady = 5)
    cancel.pack(pady = 5)
    rootwind.mainloop()
while __name__ == '__main__':
    entry()
    while exp == True:
        currentlibrary.export()

