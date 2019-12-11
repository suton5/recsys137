import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,accuracy_score,silhouette_score
from tqdm import tqdm
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

def recommend_coldstart(song_input, songs_np, songs_df, num_recommend_gmm,
        num_recommend_nn, gmm_clusters):
    '''Generates song recommendations based on Nearest Neighbours and GMM sampling.
    Inputs
    song_inputs: Index of song that user likes.
    songs_np: Numpy array of numeric attributes of dataset.
    songs_df: Full dataframe.
    num_recommend_nn: Number of songs to recommend using NN.
    num_recommend_gmm: Number of songs to recommend using GMM sampling.
    gmm_clusters: Number of clusters for GMM model. Will find optimal if specified as 0.
    Outputs
    nn_recc_songs: Recommendations using NN.
    gmm_recc_songs: Recommendations using GMM.
    '''

    query_song = songs_np[song_input]
    playlist_idx = songs_df[songs_df['track_uri'] == songs_df.iloc[song_input]['track_uri']]['pid'].values
    query_songs_df = songs_df[songs_df['pid'].isin(playlist_idx)]
    idx = query_songs_df.drop_duplicates(subset=['track_uri']).index.values
    query_songs_np = songs_np[idx]

    if gmm_clusters==0:
        #Do tuning
        print("Tuning hyperparameters for GMM.")

        n_clusters=np.arange(2, 10)
        sils=[]
        bics=[]
        iterations=20
        for n in tqdm(n_clusters):
            tmp_sil=[]
            tmp_bic=[]
            for _ in range(iterations):
                gmm=GaussianMixture(n, n_init=2).fit(query_songs_np)
                labels=gmm.predict(query_songs_np)
                sil=silhouette_score(query_songs_np, labels, metric='euclidean')
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
    label_gmm = gmm.predict(query_song.reshape(1,-1))[0]
    #to ensure we get 10 recommendations
    num_being_recommended = 0
    while num_being_recommended < num_recommend_gmm:
        samples = np.random.multivariate_normal(gmm.means_[label_gmm], gmm.covariances_[label_gmm], 2*num_recommend_gmm)
        dist, indices = nn.kneighbors(samples, n_neighbors=1)
        #drop possible duplicates
        gmm_recc = list(set(indices.flatten()))[:num_recommend_gmm]
        num_being_recommended = len(gmm_recc)

    #NN
    dist, indices = nn.kneighbors(query_song.reshape(1,-1), n_neighbors=num_recommend_nn+1)
    nn_recc = indices.flatten()[1:]

    nn_recc_songs = songs_df.iloc[idx].iloc[nn_recc]
    gmm_recc_songs = songs_df.iloc[idx].iloc[gmm_recc]
    
    return nn_recc_songs, gmm_recc_songs

if sys.argv[2] == 'Versace':
	inpt = 382

df, X_unscaled, X_train = preprocess_numeric(str(sys.argv[1]))
nn_recc_songs, gmm_recc_songs = recommend_coldstart(inpt, X_train, df, 10, 10, int(sys.argv[3]))

# NN
print('\nNN Recommendations\n')
print(nn_recc_songs[['track_name', 'artist_name']])

# NN
print('\nGMM Recommendations\n')
print(gmm_recc_songs[['track_name', 'artist_name']])


# As before, the code is commented out to avoid an error
# Builds playlists through Spotipy given a list of song recommendations

try:
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope='playlist-modify')
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=['playlist-modify', 'user-read-private'] )

spotify = spotipy.Spotify(auth=token)


new_playlist = spotify.user_playlist_create(username, 'NN Recommendations', public=True)

for i in range(nn_recc_songs.shape[0]):
	id_x = nn_recc_songs['track_uri'].values[i]
	id_x = [id_x]
	results = spotify.user_playlist_add_tracks(username, new_playlist['id'], id_x)

new_playlist = spotify.user_playlist_create(username, 'GMM Recommendations', public=True)

for i in range(gmm_recc_songs.shape[0]):
	id_x = gmm_recc_songs['track_uri'].values[i]
	id_x = [id_x]
	results = spotify.user_playlist_add_tracks(username, new_playlist['id'], id_x)

# results = spotify.user_playlist_add_tracks(username, new_playlist['id'], ['spotify:track:3zmduBNsQ6BPDTZAkXzG5K'])
