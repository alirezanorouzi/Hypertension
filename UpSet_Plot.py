#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:05:49 2021

@author: alireza
"""

import pandas as pd
from upsetplot import plot
from upsetplot import UpSet

import numpy as np
from matplotlib import pyplot as plt

path='./CombinationTest/'
fname='final_results_for_plot.csv'

df=pd.read_csv(path+fname)

values_list=['SSE','silhouette','ari_scores','count','Avg_Pr','Avg_Re','Avg_Ja','Pr_PHT','Pr_HV','Pr_PA','Pr_PPGL','Pr_CS']
for value in values_list:
#df.sort_values(ascending=True,inplace=True)

    exdf=df[['O1','O2','O3','O4','O5']]
    idx=pd.MultiIndex.from_frame(exdf)
    ex=pd.Series(np.array(df[value]), index=idx)
    upset=UpSet(ex)
    #plot(ex)
    upset.plot()
    plt.title(value)
    plt.show()
    plt.savefig(path+value)


# from upsetplot import generate_counts
# example = generate_counts()
# example  # doctest: +NORMALIZE_WHITESPACE
# cat0   cat1   cat2
# False  False  False      56
#               True      283
#        True   False    1279
#               True     5882
# True   False  False      24
#               True       90
#        True   False     429
#               True     1957

# plot(example,sort_by='value')  # doctest: +SKIP

# pyplot.show()  # doctest: +SKIP
# plot()