# -*- coding: utf-8 -*-

import streamlit  as st
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('Clusterization with K-Means and DBSCAN')

data = pd.read_csv('Data/Placas_Neltur.csv')
data = data[['X','Y']]

if st.checkbox('Show dataset'):
    st.write(data)
st.write('This dataset represents the locations of transit boards in Niterói/RJ (Brazil). The goal is to compare two clustering algorithms with spatial data.')
################ Visualization

x = data['X'].values
y = data['Y'].values

plt.figure(figsize=(10, 6))
plt.title('Transit boards of Niterói', fontsize=15)
plt.xlabel('latitude', fontsize=12)
plt.ylabel('longitude', fontsize=12)
plt.scatter(x, y,   alpha=0.5 )
st.pyplot()
########################

#K-means
st.markdown("<h1 style='text-align: center; color: red;'>K-Means</h1>", unsafe_allow_html=True)

k_values = np.arange(2,10)
st.write('Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group.')
st.markdown('* k : The number of clusters to form as well as the number of centroids to generate. ')
k_select = st.selectbox('Select the value of K', k_values)
KM_clusters = KMeans(n_clusters=k_select, init='k-means++').fit(data) # initialise and fit K-Means model

KM_clustered = data.copy()
KM_clustered.loc[:,'Cluster'] = KM_clusters.labels_ # append labels to points

sns.scatterplot(data=KM_clustered,hue='Cluster', x="X", y="Y", palette='Set1').set(title='Clusterization with K-Means')
st.pyplot()

################

#DBSCAN
st.markdown("<h1 style='text-align: center; color: red;'>DBSCAN</h1>", unsafe_allow_html=True)
st.write('DBSCAN groups together points that are close to each other based on a distance measurement (usually Euclidean distance) and a minimum number of points. It also marks as outliers the points that are in low-density regions.')
min_samples = np.arange(2,10)  # min_samples values to be investigated
eps_values = np.arange(0.00015,0.00023,1.6000000000000003e-05) # eps values to be investigated

st.markdown('* eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function. \n * min_samples:The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.')
eps = st.selectbox('Select the value of eps', eps_values)
min = st.selectbox('Select the number of minimum samples', min_samples)


data2 = np.radians(data) #necessary for the harversine method
DBS_clustering = DBSCAN(eps= eps  , min_samples=min, metric='haversine',algorithm='ball_tree').fit(data2)  #maior que 0,01 nao gera cluster

DBSCAN_clustered = data2.copy()
DBSCAN_clustered.loc[:,'Cluster'] = DBS_clustering.labels_ # append labels to points

sns.scatterplot(data=DBSCAN_clustered,hue='Cluster', x="X", y="Y", palette='Set1').set(title='Clusterization with DBSCAN')
st.pyplot()
##############

st.markdown('References: \n * https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a \n * https://towardsdatascience.com/how-dbscan-works-and-why-should-i-use-it-443b4a191c80 \n * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html \n * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html')