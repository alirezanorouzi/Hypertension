#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:29:43 2021

@author: alireza
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:44:34 2021

@author: alireza
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def FeildLableEncoder(fld):
    lb_make = LabelEncoder() 
    df[fld+'bl'] = lb_make.fit_transform(df[fld]) 
    return
def NormFld(fld):
#    if fld in NormList:
#        zscore=10*( x-x.min() )/ x.max()-x.min()
#        df.groupby("_month")[fld].transform(lambda x: zscore)         
#    print(fld,df[fld].min(),df[fld].max())        
    df[fld]=( df[fld]-df[fld].min() )/ ( df[fld].max()-df[fld].min() )
#    df[fld]=df[fld]/df[fld].max()
    return


#main
path='ensat_ht_Pmirna_all_patients_5_stable_mirna_without_outliers_Alireza.csv'
df_org=pd.read_csv(path)

df=df_org[:][:]
df=df.drop(columns=['Patient_ID'])

#k=2
#df.loc[(df['Condition'] != 'HV'), 'Condition'] = str('NO')
#df.loc[(df['Condition'] != 'PHT'), 'Condition'] = str('NO')
#df.loc[(df['Condition'] != 'PA'), 'Condition'] = str('NO')
#df.loc[(df['Condition'] != 'PPGL'), 'Condition'] = str('NO')
#df.loc[(df['Condition'] != 'CS'), 'Condition'] = str('NO')

x=df[:][:]

for fld in df.columns:
    if (pd.api.types.is_string_dtype(df[fld])):
        #df[fld].replace(np.nan,'other',inplace=True)
        FeildLableEncoder(fld)
        x[fld]=df[fld+'bl']
        



label=df[:]['Conditionbl']


#Scaling and Feature transferring
for fld in df.columns:
    if (pd.api.types.is_integer_dtype(df[fld])):
        #df[fld].replace(np.nan,0,inplace=True)        
        #df[fld] = df.groupby("_Customer_Rank")[fld].transform(lambda x: x.fillna(x.mean()))
        NormFld(fld)
        x[fld]=df[fld]


for fld in df.columns:
    if (pd.api.types.is_float_dtype(df[fld])):
        #df[fld] = df.groupby("_Customer_Rank")[fld].transform(lambda x: x.fillna(x.mean()))
        #df[fld].replace(np.nan,0,inplace=True)
        NormFld(fld)
        x[fld]=df[fld]
        
        

#x=x.drop(columns=['Patient_ID'])
x=x.drop(columns=['CentreID']) #fas considrable effect on clustering
x=x.drop(columns=['Gender']) #has low  effect on clustering
#x=x.drop(columns=['Age'])    
x=x.drop(columns=['AoS']) #dosent have greate effect on clustering, When exist Silhote goes to 1
x=x.drop(columns=['VH_Present'])
x=x.drop(columns=['Condition'])

x=x.drop(columns=['CentreIDbl']) #fas considrable effect on clustering
x=x.drop(columns=['Genderbl']) #has low  effect on clustering
x=x.drop(columns=['Conditionbl'])
x=x.drop(columns=['VH_Presentbl'])


# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(x)
# x=scaled_features
# #x=x.drop(columns=['AoS'])


# #----------------------------PCA
# from sklearn.decomposition import PCA

# pca = PCA(n_components=10)
# pca.fit(x)  
# x=pca.transform(x)



# #----------------------------Upsampling
# from sklearn.utils import resample
# # Separate majority and minority classes
# df_majority = x[df['Condition']=='NO']
# df_minority = x[df['Condition']!='NO']
 
# # Upsample minority class
# df_minority_upsampled = resample(df_minority, 
#                                   replace=True,                                  # sample with replacement
#                                   n_samples=(df['Condition']!='NO').count(),     # to match majority class
#                                   random_state=123)                              # reproducible results
# df_minority_upsampled_y=pd.DataFrame([0]*(df['Condition']!='NO').count())
# df_majority_y=label[df['Condition']=='NO']
# # Combine majority class with upsampled minority class
# df_upsampled = pd.concat([df_majority, df_minority_upsampled])
# x=df_upsampled
# label= pd.concat([df_majority_y, df_minority_upsampled_y])

# #-----------------------------------ICA
# from sklearn.datasets import load_digits
# from sklearn.decomposition import FastICA

# transformer = FastICA(n_components=7,
#         random_state=0)
# X_transformed = transformer.fit_transform(x)
# x=X_transformed

#--------------------------------------------------------
# from sklearn.decomposition import PCA

# pca = PCA(n_components=10)
# pca.fit(x)  
# x2=pca.transform(x)

#----------------------------LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=4)
x_lda = lda.fit_transform(x, label)
#X_test = lda.transform(X_test)

# '''
# Classification part
# '''
# if 'AoS' in x.columns:
#     x=x.drop(columns=['AoS'])      
        
x2=x_lda
y=label
#--------------- importance feature
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

#kf=KFold(n_splits=7,random_state=0)

#Train Different Model
models=[]
models.append(('SVC',SVC()))
models.append(('LR', LogisticRegression()))
models.append(('Knn', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('DT', DecisionTreeClassifier()))
#models.append(('GBC', GradientBoostingClassifier(n_estimators=30, learning_rate=.5, max_features=2, max_depth=2, random_state=0)))
#models.append(('NN', MLPClassifier(solver='lbfgs', activation='relu',hidden_layer_sizes=(2, ), random_state=1) ) )

results=[]
name=[]
p=pd.DataFrame()
p['label']=label

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
    
for name,model in models:
    print(name)
    kfold=KFold(n_splits=10,random_state=1)


    acc=cross_val_score(model,x2,y,cv=kfold,scoring='accuracy').mean() 
#    pre=cross_val_score(model,x2,y,cv=kfold,scoring='precision').mean()
#    rec=cross_val_score(model,x2,y,cv=kfold,scoring='recall').mean()
    pre=0   
    rec=0
    print ("accuracy : ", acc    )
    print ("precision : ", pre    )
    print ("recall : ", rec    )
    results.append((name,acc,pre,rec,model))
    model.fit(x2,y)

    y_pred = model.predict(x2)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y, y_pred, target_names=target_names))
# #-------------------------------------------------
# #aa=aa.sort_values('pr',ascending=False)    
