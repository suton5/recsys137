import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,accuracy_score,silhouette_score
from tqdm import tqdm
import tensorflow as tf
import sys
import spotipy
import spotipy.util as util
import os
import sys
from json.decoder import JSONDecodeError

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

def SelBest(arr, N):
    '''Return the N smallest values.'''
    idx=np.argsort(arr)[:N]
    return arr[idx]

def recommend_mood(mood_input, nn_model, songs_np, songs_df, num_recommend_sample, num_recommend_nn):
    '''Generates song recommendations based on Nearest Neighbours and GMM sampling.

    Inputs

    mood_inputs: Mood of the songs that the user wants.
    nn_model: The neural network model mapping features to mood.
    songs_np: Numpy array of numeric attributes of dataset.
    songs_df: Full dataframe.
    num_recommend_sample: Number of songs to recommend using sample method.
    num_recommend_nn: Number of songs to recommend using NN.

    Outputs
    
    sample_recc_songs: Recommendations using sample method.
    nn_recc_songs: Recommendations using neural network method.
    '''
    
    #Sample method
    
    sample_playlist = pd.read_csv('../test_data/'+mood_input+'.csv')
    sample_song_size = 0
    while sample_song_size == 0:
        #keep resampling till we get a song in our playlist
        sample_idx = np.random.randint(low=0, high=sample_playlist.shape[0], size=1)[0]
        sample_trackuri = sample_playlist.iloc[sample_idx]['track_uri']
        sample_song_idx = songs_df[songs_df['track_uri'] == sample_trackuri].index
        sample_song_size = sample_song_idx.size
    song_input = sample_song_idx[0]

    query_song = songs_np[song_input]
    playlist_idx = songs_df[songs_df['track_uri'] == songs_df.iloc[song_input]['track_uri']]['pid'].values
    query_songs_df = songs_df[songs_df['pid'].isin(playlist_idx)]
    idx = query_songs_df.drop_duplicates(subset=['track_uri']).index.values
    query_songs_np = songs_np[idx]

    nn = NearestNeighbors().fit(query_songs_np)
    dist, indices = nn.kneighbors(query_song.reshape(1,-1), n_neighbors=num_recommend_sample+1)
    nn_recc = indices.flatten()[1:]

    sample_recc_songs = songs_df.iloc[idx].iloc[nn_recc]
    
    #Neural Network

    idx = songs_df.drop_duplicates(subset=['track_uri']).index.values
    nodup_songs_np = songs_np[idx]
    y_pred = nn_model.predict_classes(nodup_songs_np)
    if mood_input == 'angry':
        check = 0, 
    elif mood_input == 'party':
        check = 1, 
    elif mood_input == 'sad':
        check = 2
    else:
        check = 3

    rec_idx_all = np.argwhere(y_pred==check)
    np.random.shuffle(rec_idx_all)
    rec_idx = rec_idx_all[:num_recommend_nn]
    
    nn_recc_songs = songs_df.iloc[idx].iloc[rec_idx.flatten()]
    
    return sample_recc_songs, nn_recc_songs

model = tf.keras.models.load_model('model.h5')
print("Loaded neural network model from disk")

df, X_unscaled, X_train = preprocess_numeric(str(sys.argv[1]))
sample_recc_songs, nn_recc_songs = recommend_mood(str(sys.argv[2]), model, X_train, df, 10, 10)

# Sample
print('\nSample-based Recommendations\n')
print(sample_recc_songs[['track_name', 'artist_name']])

# NN
print('\nNN-based Recommendations\n')
print(nn_recc_songs[['track_name', 'artist_name']])


# As before, the code is commented out to avoid an error
# Builds playlists through Spotipy given a list of song recommendations

try:
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope='playlist-modify')
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=['playlist-modify', 'user-read-private'] )

spotify = spotipy.Spotify(auth=token)


new_playlist = spotify.user_playlist_create(username, 'Sample-based Recommendations', public=True)

for i in range(sample_recc_songs.shape[0]):
	id_x = sample_recc_songs['track_uri'].values[i]
	id_x = [id_x]
	results = spotify.user_playlist_add_tracks(username, new_playlist['id'], id_x)

new_playlist = spotify.user_playlist_create(username, 'NN-based Recommendations', public=True)

for i in range(nn_recc_songs.shape[0]):
	id_x = nn_recc_songs['track_uri'].values[i]
	id_x = [id_x]
	results = spotify.user_playlist_add_tracks(username, new_playlist['id'], id_x)

# results = spotify.user_playlist_add_tracks(username, new_playlist['id'], ['spotify:track:3zmduBNsQ6BPDTZAkXzG5K'])
