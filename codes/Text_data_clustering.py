# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:10:50 2021

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as hc
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

##Reference:
    ## https://scikit-learn.org/stable/modules/clustering.html
## https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
    
#bring in the data
path = "D:/GU/School Work/Fall 21/ANLY 501/assignment 1/cleaned data/"
filename="NewsData_cleaned_101321.csv"

smalldata=pd.read_csv(path+filename)
print(type(smalldata))
print(smalldata)

mydata = smalldata.drop(columns = "LABEL")
print(mydata)

#####
# KMEANS
# Use k-means clustering on the data.

# Create clusters 
k = 4
## Sklearn required you to instantiate first
kmeans = KMeans(n_clusters=k)
kmeans.fit(mydata)   ## run kmeans

labels = kmeans.labels_
print(labels)

centroids = kmeans.cluster_centers_
print(centroids)

prediction = kmeans.predict(mydata)
print(prediction)


##########################################
##
## Try to predict new data vectors
##
################################################
NewData=[0, 3, 4, 0, 1, 2, 3, 0, 0, 1, 0]
#NewData=[3.6, 90, 169, 7, 91]
print(type(NewData))
NewData=np.asarray(NewData)
#print(NewData)
print(NewData.shape)
NewData=NewData.reshape(1,-1)
print(NewData.shape)

#NewData=np.transpose(NewData)
#print(NewData)

#print(kmeans.predict([[3.6, 90, 169, 7, 91]]))
#print(kmeans.predict([[2.8, 70, 139, 2, 61]]))
#print(kmeans.predict(NewData))

print(kmeans.predict(NewData))
print(kmeans.predict([[4, 3, 0, 0, 1, 0, 3, 7, 0, 0, 0]]))
print(kmeans.predict([[0, 0, 4, 6, 1, 0, 1, 0, 0, 0, 0]]))
print(kmeans.predict([[0, 0, 1, 6, 6, 7, 8, 0, 1, 0, 1]]))
print(kmeans.predict([[0, 0, 1, 1, 0, 1, 1, 0, 4, 4, 5]]))


################################################
##
##         Look at best values for k
##
###################################################

## Elbow method
SS_dist = []

values_for_k=range(2,9)
#print(values_for_k)

for k_val in values_for_k:
    print(k_val)
    k_means = KMeans(n_clusters=k_val)
    model = k_means.fit(mydata)
    SS_dist.append(k_means.inertia_)
    
print(SS_dist)
print(values_for_k)

plt.plot(values_for_k, SS_dist, 'bx-')
plt.xlabel('value')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal k Choice')
plt.show()


## Silhouette & Calinski
Sih=[]
Cal=[]
k_range=range(2,8)

for k in k_range:
    k_means_n = KMeans(n_clusters=k)
    model = k_means_n.fit(mydata)
    Pred = k_means_n.predict(mydata)
    labels_n = k_means_n.labels_
    R1=metrics.silhouette_score(mydata, labels_n, metric = 'euclidean')
    R2=metrics.calinski_harabasz_score(mydata, labels_n)
    Sih.append(R1)
    Cal.append(R2)

print(Sih) ## higher is better
print(Cal) ## higher is better

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(k_range,Sih)
ax1.set_title("Silhouette")
ax1.set_xlabel("")
ax2.plot(k_range,Cal)
ax2.set_title("Calinski_Harabasz_Score")
ax2.set_xlabel("k values")

####################################################
##
##      Look at Clusters
##
####################################################

 

# It is often best to normalize the data 
## before applying the fit method
## There are many normalization options
## This is an example of using the z score
smalldata_normalized=(mydata - mydata.mean()) / mydata.std()
print(smalldata_normalized)
##################################################
## PCA - 
## or principle component analysis
## can be used to identify the principle comp
## the vectors (columns) with the highest eigenvalue
## or most distinct/largest variation from the other components
## This is a method of dimensionality reduction
#######################################################
print(smalldata_normalized.shape[0])   ## num rows
print(smalldata_normalized.shape[1])   ## num cols

NumCols=smalldata_normalized.shape[1]

## Instantiated my own copy of PCA
My_pca = PCA(n_components=2)  ## I want the two prin columns

## Transpose it
smalldata_normalized=np.transpose(smalldata_normalized)
My_pca.fit(smalldata_normalized)

print(My_pca)
print(My_pca.components_.T)
KnownLabels=smalldata['LABEL']

# Reformat and view results
Comps = pd.DataFrame(My_pca.components_.T,
                        columns=['PC%s' % _ for _ in range(2)],
                        index=smalldata_normalized.columns
                        )
print(Comps)
print(Comps.iloc[:,0])
#RowNames = list(Comps.index)
#print(RowNames)

########################
## Look at 2D PCA clusters
############################################

plt.figure(figsize=(12,12))
plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], s=100, color="green")

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Scatter Plot Clusters PC 1 and 2",fontsize=15)
for i, label in enumerate(KnownLabels):
    #print(i)
    #print(label)
    plt.annotate(label, (Comps.iloc[i,0], Comps.iloc[i,1]))

plt.show()


###############################################
##
##         DBSCAN
##
###############################################


MyDBSCAN = DBSCAN(eps=5.6, min_samples=1)
## eps:
    ## The maximum distance between two samples for 
    ##one to be considered as in the neighborhood of the other.
MyDBSCAN.fit_predict(mydata)
print(MyDBSCAN.labels_)

#########################################
##
##  Hierarchical 
##
#########################################


MyHC = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
FIT=MyHC.fit(mydata)
HC_labels = MyHC.labels_
print(HC_labels)


plt.figure(figsize =(12, 12))
plt.title('Hierarchical Clustering')
dendro = hc.dendrogram((hc.linkage(mydata, method ='ward')))

## WARD
## Recursively merges the pair of clusters that 
## minimally increases within-cluster variance.

from sklearn.metrics.pairwise import euclidean_distances
EDist=euclidean_distances(mydata)
print(EDist)