import librosa
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
# if you are not running this script in a notebook environment, you may need to import
# perform_mann_whitney_and_permutation_test from the pitch script.

def get_spectral_feature_data(file_path):
    '''
    this function is responsible for getting the spectral centroid, bandwidth
    , flatness and rolloff. It follows methodology demonstrated in notebook found in :
    imsparsh. (2015)
    and 
    McFee et al. (2015) 
    '''
    y, sr = librosa.load(file_path, sr=None)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return np.mean(spectral_centroid), np.mean(spectral_bandwidth), np.mean(spectral_flatness), np.mean(spectral_rolloff)

def process_directory(directory):
    features = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            features.append(get_spectral_feature_data(file_path))
    return features


# this dataframe will hold all the accent feature data for analysis
accent_features_df = pd.DataFrame()

# iterate through utterances and get spectral features for each utterance in
# the base directory
for utterance in utterances:
    for accent, base_dir in base_directories.items():
        directory = os.path.join(base_dir, utterance, utterance) 
        features = process_directory(directory)
        df = pd.DataFrame(features, columns=['spectral centroid', 'spectral bandwidth', 
                                             'spectral flatness', 'spectral roll-off'])
        df['Accent'] = accent
        df['Utterance'] = utterance
        accent_features_df = pd.concat([accent_features_df, df])

# stats output and mann whitney u and permutation testing
for utterance in utterances:
    print(" ")
    print(f"stats output for utterance: '{utterance}'")
    for feature in ['spectral centroid', 'spectral bandwidth', 'spectral flatness', 'spectral roll-off']:
        print(" ")
        print(f"processing feature: {feature}")
        accents = accent_features_df['Accent'].unique()
        for i in range(len(accents) - 1):
            for x in range(i + 1, len(accents)):
                accent_1 = accents[i]
                accent_2 = accents[x]
                accent_feature_data_1 = accent_features_df[(accent_features_df['Accent'] == accent_1) & 
                                         (accent_features_df['Utterance'] == utterance)][feature]
                accent_feature_data_2 = accent_features_df[(accent_features_df['Accent'] == accent_2) & 
                                         (accent_features_df['Utterance'] == utterance)][feature]
                stat, p_value, p_value_perm = perform_mann_whitney_and_permutation_test(
                    accent_feature_data_1, accent_feature_data_2)
                print(f"mann whitney u and perm test on {accent_1} and {accent_2} with feature -{feature}: U={stat}, p-value={p_value}, Permutation p-value={p_value_perm}")
