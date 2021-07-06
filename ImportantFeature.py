#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:16:25 2021

@author: alireza
"""

# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import pandas as pd
from sklearn.linear_model import LogisticRegression

path='CleanedData2.csv'
x=pd.read_csv(path)

label=x['Condition']
y=x['ConditionID']
x=x.drop(columns=['Unnamed: 0'])
x=x.drop(columns=['Condition'])
x=x.drop(columns=['ConditionID'])
#--------------------------
from sklearn.feature_selection import SelectKBest, f_classif
fs= SelectKBest(f_classif, k=6) #6
x_best = fs.fit_transform(x,y) 

mask = fs.get_support() #list of booleans
new_features = [] # The list of your K best features
feature_names = list(x.columns.values)
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
#--------------------------
model = LinearRegression()
# fit the model
model.fit(x, y)
# get importance
importance = model.coef_
# summarize feature importance
importance=importance.reshape(174,1)
importance=importance.squeeze()
for i,v in enumerate(importance):
	print((i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


#---------------------------
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
model = DecisionTreeRegressor()
# fit the model
model.fit(x, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()



#------------------------
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot

model = RandomForestRegressor()
# fit the model
model.fit(x, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

#----------------------------
import plotly.express as px

fig = px.scatter_3d(x, x="miRNA_48", y="miRNA_4", z="miRNA_8", color='Condition')
fig.show()
