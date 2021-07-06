
import numpy as np; 
import seaborn as sns; 
import pandas as pd


path='./CombinationTest/'
fname='final_results_for_plot.csv'

df=pd.read_csv(path+fname)

dfplt=df[:][:]

dfplt['Omics']=dfplt['Omics'].str.replace('Omics_', '')
dfplt['Omics']=dfplt['Omics'].str.replace('_', ',')
dfplt['Omics']=dfplt['Omics'].str.replace('.csv', '')

dfplt['len']= dfplt['Omics'].apply(len)
dfplt=dfplt.sort_values(['len','Omics'])

for fld in dfplt.columns:
    if (pd.api.types.is_integer_dtype(dfplt[fld])) or (pd.api.types.is_float_dtype(dfplt[fld])):
        dfplt[fld]=100*( dfplt[fld]-dfplt[fld].min() )/ ( dfplt[fld].max()-dfplt[fld].min() )
cols=dfplt['Omics']
cols=cols.to_list()
df_heat=dfplt[['SSE','silhouette','ari_scores','count','Avg_Pr','Avg_Re','Avg_Ja','Pr_PHT','Pr_HV','Pr_PA','Pr_PPGL','Pr_CS']].T
df_heat.columns=cols

ax = sns.heatmap(df_heat)
