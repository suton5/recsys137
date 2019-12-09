# RecSys137

1. Feature Extraction
```
python feature_extract.py start end
```
Extracts song features by using the Spotify API for files `start` to `end` and create modified CSVs.

2. Baseline Recommendations
```
python recommend_baseline.py file song_idx
```
Generates song recommendations for `song_idx` from `file`.
