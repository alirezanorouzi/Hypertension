#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:14:30 2021

@author: alireza
"""
import pandas as pd
import os

l=[]
path='./CombinationTest/'
files = os.listdir(path)
for i,name in enumerate(files):
    print(i,':',name)
    if (name.find('.csv')>=0) & (name.find('Omics')>=0):
        print('Uploaded sucessfully: ',name)
        df=pd.read_csv(path+name)
        print([name, df['sse'][0], df['silhouette'][0], df['ari_scores'][0]])
        PrList=[]
        for j,il in enumerate(['PHT','HV','PA','PPGL','CS']):
            a=df.loc[(df['Pr']>0) & (df['Condition']==il),'Pr']
            if a.empty:
                PrList.append(0)
            else:
                PrList.append(a.tolist()[0])
        l.append([name, df['sse'][0], df['silhouette'][0], df['ari_scores'][0],df.count()[0],
                df['Avg_Pr'][0],df['Avg_Re'][0],df['Avg_Ja'][0]]+PrList)
        
df_omics=pd.DataFrame(l)

df_omics.columns=('Omics', 'SSE','silhouette','ari_scores','count','Avg_Pr','Avg_Re','Avg_Ja','Pr_PHT','Pr_HV','Pr_PA','Pr_PPGL','Pr_CS')

for i in range(1,6):
    O_N='O'+str(i)
    df_omics[O_N] =list(map(lambda x: O_N in x, df_omics['Omics']))

df_omics.to_csv(path+'final_results_for_plot.csv')
    
for


from matplotlib import pyplot

pyplot.bar( [x for x in range( df.count()[0] ) ], df_omics['silhouette'] )
pyplot.show()



pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()