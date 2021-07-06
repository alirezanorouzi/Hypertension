#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:50:05 2021

@author: alireza
"""

from sklearn.cluster import AffinityPropagation
import numpy as np
import pandas as pd

X = x
clustering = AffinityPropagation(random_state=1).fit(X)
clustering

cl=pd.DataFrame(clustering.labels_)


l=pd.DataFrame({'AffinityPropagation':clustering.labels_})
l['ConditionID']=label
l['Condition']=df_org[:][['Condition']]



from sklearn.cluster import SpectralClustering
import numpy as np
X = x
clustering = SpectralClustering(n_clusters=5,
        assign_labels='discretize',
        random_state=0).fit(X)
clustering.labels_

clustering

l=pd.DataFrame({'SpectralClustering':clustering.labels_})
l['ConditionID']=label
l['Condition']=df_org[:][['Condition']]

 vc=l.value_counts()



from sklearn.cluster import AgglomerativeClustering
import numpy as np
X = x
clustering = AgglomerativeClustering(n_clusters=5).fit(X)
clustering

clustering.labels_

l=pd.DataFrame({'AgglomerativeClustering':clustering.labels_})
l['ConditionID']=label
l['Condition']=df_org[:][['Condition']]

 vc=l.value_counts()
