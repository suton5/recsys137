import spotipy
import spotipy.util as util
from json.decoder import JSONDecodeError
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import sys
import os

# This is for Jovin's spotify account
username='1163796537'
client_id='cd2e6f4c5d0c4981b8914e574582d28d'
client_secret='9c6180a30f2b4ca897455989199d0af2'
redirect_uri='https://google.com'

def preprocess_numeric(path):
    '''Read in songs CSV and output Numpy array of the numeric columns after standardising.'''
    df = pd.read_csv(path)
    df_numeric = df[['danceability','energy','loudness','speechiness','acousticness','instrumentalness',\
        'liveness','valence','tempo']]
#     df_numeric = df[['danceability', 'instrumentalness','valence']]
    X_scaled = StandardScaler().fit_transform(df_numeric)
    
    return df, np.array(df_numeric), X_scaled

def recommend(song_input, songs_np, songs_df, num_recommend_gmm, num_recommend_nn, gmm_clusters=8):
    '''Generates song recommendations based on Nearest Neighbours and GMM sampling.
    
    Inputs
    
    song_inputs: Index of song that user likes.
    songs_np: Numpy array of numeric attributes of dataset.
    songs_df: Full dataframe
    
    num_recommend_nn: Number of songs to recommend using NN.
    num_recommend_gmm: Number of songs to recommend using GMM sampling.
    gmm_clusters: Number of clusters for GMM.
    
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
    
    gmm = GaussianMixture(n_components=gmm_clusters).fit(query_songs_np)
    nn = NearestNeighbors().fit(query_songs_np)
    
    #GMM sampling
    label_gmm = gmm.predict(query_song.reshape(1,-1))
    samples = np.random.multivariate_normal(gmm.means_[0], gmm.covariances_[0], num_recommend_gmm)
    dist, indices = nn.kneighbors(samples, n_neighbors=1)
    gmm_recc = indices.flatten()
        
    #NN
    dist, indices = nn.kneighbors(query_song.reshape(1,-1), n_neighbors=num_recommend_nn+1)
    nn_recc = indices[:,1:][0]
    
#     return gmm_recc, nn_recc
    return idx, nn_recc, gmm_recc

df, X_unscaled, X_train = preprocess_numeric(str(sys.argv[1]))
idx, nn_recc, gmm_recc = recommend(int(sys.argv[2]), X_train, df, 10, 10)


# GMM
print('GMM\n')
gmm_list = df.iloc[idx].iloc[gmm_recc][['track_uri']]
print(df.iloc[idx].iloc[gmm_recc][['track_name', 'artist_name']])

# NN
print('\nNN\n')
nn_list = df.iloc[idx].iloc[nn_recc][['track_uri']]
print(df.iloc[idx].iloc[nn_recc][['track_name', 'artist_name']])

playist_elements = list(nn_list['track_uri']) + list(gmm_list['track_uri'])

try:
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope='playlist-modify')
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=['playlist-modify', 'user-read-private'] )

spotify = spotipy.Spotify(auth=token)

new_playlist = spotify.user_playlist_create(username, 'Generated Playlist RecSys 137', public=True)
results = spotify.user_playlist_add_tracks(username, new_playlist['id'], playist_elements)