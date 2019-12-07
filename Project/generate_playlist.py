import spotipy
import spotipy.util as util
import pandas as pd
import numpy as np
import os
import sys
from json.decoder import JSONDecodeError



# username = '1218319361'
# client_id = '25562ed2e5c14bf4931131f19369dc96'
# client_secret = '41a147b3152d47d59489f74a5e5eca38'
# redirect_uri='https://google.com'

username='1163796537'
client_id='cd2e6f4c5d0c4981b8914e574582d28d'
client_secret='9c6180a30f2b4ca897455989199d0af2'
redirect_uri='https://google.com'

try:
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope='playlist-modify')
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=['playlist-modify', 'user-read-private'] )

spotify = spotipy.Spotify(auth=token)


new_playlist = spotify.user_playlist_create(username, 'Spotipy Test', public=True)

results = spotify.user_playlist_add_tracks(username, new_playlist['id'], ['spotify:track:3zmduBNsQ6BPDTZAkXzG5K'])
