import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tqdm import tqdm
import sys

def preprocess_numeric(path):
    '''Read in songs CSV and output Numpy array of the numeric columns after standardising.'''
    df = pd.read_csv(path)
    df_numeric = df[['danceability','energy','loudness','speechiness','acousticness','instrumentalness',\
        'liveness','valence','tempo']]
#     df_numeric = df[['danceability', 'instrumentalness','valence']]
    X_scaled = StandardScaler().fit_transform(df_numeric)

    return df, np.array(df_numeric), X_scaled

def SelBest(arr, N):
    '''Return the N smallest values.'''
    idx=np.argsort(arr)[:N]
    return arr[idx]

def recommend(song_input, songs_np, songs_df, num_recommend_gmm,
        num_recommend_nn, gmm_clusters):
    '''Generates song recommendations based on Nearest Neighbours and GMM sampling.

    Inputs

    song_inputs: Index of song that user likes.
    songs_np: Numpy array of numeric attributes of dataset.
    songs_df: Full dataframe

    num_recommend_nn: Number of songs to recommend using NN.
    num_recommend_gmm: Number of songs to recommend using GMM sampling.
    gmm_clusters: Number of clusters for GMM model. Will find optimal if none specified.

    Outputs

    idx: Indices corresponding to data subset for playlists that contain query song.
    gmm_recc: num_recommend_gmm Numpy array containing indices of the recommended songs using GMM.
    nn_recc: num_recommend_nn Numpy array containing indices of the recommended songs using NN.
    '''

    query_song = songs_np[song_input]
    playlist_idx = songs_df[songs_df['track_uri'] == songs_df.iloc[song_input]['track_uri']]['pid'].values
    query_songs_df = songs_df[songs_df['pid'].isin(playlist_idx)]
    idx = query_songs_df.drop_duplicates(subset=['track_uri']).index.values
    query_songs_np = songs_np[idx]

    if gmm_clusters==0:
        #Do tuning
        print("Tuning hyperparameters for GMM.")

        n_clusters=np.arange(2, 20)
        sils=[]
        bics=[]
        iterations=20
        for n in tqdm(n_clusters):
            tmp_sil=[]
            tmp_bic=[]
            for _ in range(iterations):
                gmm=GaussianMixture(n, n_init=2).fit(query_songs_np)
                labels=gmm.predict(query_songs_np)
                sil=metrics.silhouette_score(query_songs_np, labels, metric='euclidean')
                tmp_sil.append(sil)
                tmp_bic.append(gmm.bic(query_songs_np))
            val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
            sils.append(val)
            val=np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))
            bics.append(val)
        gmm_clusters = int((n_clusters[np.argmin(bics)] + n_clusters[np.argmax(sils)])/2)

        print("Optimal number of clusters: {}.".format(gmm_clusters))

    print("Fitting models.")

    gmm = GaussianMixture(n_components=gmm_clusters).fit(query_songs_np)
    nn = NearestNeighbors().fit(query_songs_np)

    print("Generating recommendations.")

    #GMM sampling
    label_gmm = gmm.predict(query_song.reshape(1,-1))
    samples = np.random.multivariate_normal(gmm.means_[0], gmm.covariances_[0], num_recommend_gmm)
    dist, indices = nn.kneighbors(samples, n_neighbors=1)
    gmm_recc = indices.flatten()

    #NN
    dist, indices = nn.kneighbors(query_song.reshape(1,-1), n_neighbors=num_recommend_nn+1)
    nn_recc = indices[:,1:][0]

    return idx, nn_recc, gmm_recc

df, X_unscaled, X_train = preprocess_numeric(str(sys.argv[1]))
idx, nn_recc, gmm_recc = recommend(int(sys.argv[2]), X_train, df, 10, 10, int(sys.argv[3]))

# GMM
print('\nGMM Recommendations\n')
print(df.iloc[idx].iloc[gmm_recc][['track_name', 'artist_name']])

# NN
print('\nNN Recommendations\n')
print(df.iloc[idx].iloc[nn_recc][['track_name', 'artist_name']])
