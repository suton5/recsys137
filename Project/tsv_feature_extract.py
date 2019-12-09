import spotipy
import spotipy.util as util
import pandas as pd
import numpy as np
import os
import sys
from json.decoder import JSONDecodeError

username='<hidden>'
client_id='<hidden>'
client_secret='<hidden>'
redirect_uri='https://google.com'

try:
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)

spotify = spotipy.Spotify(auth=token)

# Input TSV with songs to be parsed
df1 = pd.read_csv("sad.tsv", delimiter='\t')

df1.dropna(inplace=True)

attrs=['danceability',
  'energy',
  'key',
  'loudness',
  'mode',
  'speechiness',
  'acousticness',
  'instrumentalness',
  'liveness',
  'valence',
  'tempo',
  'type',
  'id',
  'uri',
  'track_href',
  'analysis_url',
  'duration_ms',
  'time_signature']

for key in attrs:
    df1[key]=0

i=0
j=99
ender=0
error=0

while ender==0:
    if j>=df1.shape[0]:
        j=df1.shape[0]
        ender=1
    try:
        output = spotify.audio_features(tracks=df1.loc[i:j]['track_uri'].values)
        for key in output[0]:
            df1.loc[i:j, key]=np.array([d[key] for d in output])
        i=j+1
        j+=100

    except TypeError:
        error=1
        break

if error==0:  
	# Export to CSV; rename!!!!
    export_csv = df1.to_csv ('sad.csv')
