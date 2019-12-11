#!/usr/bin/env python3


# Script is used to generate plots for our website


import pandas as pd
import numpy as np
import scipy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import spotipy
import spotipy.util as util

def preprocess_numeric(path):
    '''Read in songs CSV and output Numpy array of the numeric columns after standardising.'''
    df = pd.read_csv(path)
    df_numeric = df[['danceability','energy','loudness','speechiness','acousticness','instrumentalness',\
        'liveness','valence','tempo']]
    X_scaled = StandardScaler().fit_transform(df_numeric)
    
    return df, np.array(df_numeric), X_scaled

df, X_unscaled, X_train = preprocess_numeric('../test_data/songs620.csv')

# Scatter matrix 1
cor_columns = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness',\
        'liveness','valence','tempo']
scatter_matrix(df[cor_columns], figsize=(40, 40), c="#1ED761")
plt.suptitle("Scatter matrix for all track feature attributes obtained from the Spotify API")
plt.show()

# PCA
pca=PCA().fit(X_train)
pca_X_train=pca.transform(X_train)

# PCA Scatter matrix
fig, ax=plt.subplots(9,9, figsize=(40,40))
for i in range(9):
    for j in range(9):
        ax[i,j].scatter(pca_X_train[:,i], pca_X_train[:,j], alpha=0.1, c="#1ED761")

plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0).fit(PCA().fit_transform(X_unscaled))
pd.value_counts(kmeans.labels_)

# Kmeans plot
plt.figure(figsize=(12,8))
plt.scatter(PCA().fit_transform(X_unscaled)[:,0], PCA().fit_transform(X_unscaled)[:,1], alpha=0.1, c="#1ED761")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', s=50)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('Scatter plot of first two PCA components', fontsize=15)
plt.show()