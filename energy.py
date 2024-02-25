import os
import librosa
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import scipy.stats as stats
# if you are not running this script in a notebook environment, you may need to import
# perform_mann_whitney_and_permutation_test from the pitch script.

def get_energy_features(file_path):
    '''
    return the count of peaks and average energy of these peaks. This 
    code was guided by work completed in:
    iranroman (2018)
    and 
    Plotly Technologies Inc (2015)
    '''
    y, sr = librosa.load(file_path, sr=None)
    frame_length = 1024
    hop_length = 512
    energy = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)])
    peaks, _ = find_peaks(energy, height=np.mean(energy))
    num_peaks = len(peaks)
    mean_energy = np.mean(energy[peaks]) if peaks.size > 0 else 0
    return {"num_peaks": num_peaks, "mean_energy": mean_energy}

def analyse_directory(directory):
    # iterate over the directory containing wav files and process each one
    data = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            features = get_energy_features(file_path)
            data.append(features)
    return pd.DataFrame(data)

# changing the utterance list and base directories since the focus is now on constonants
utterances = ['d', 't']
base_directories = {
    'Dublin': '/home/garrett/Workspace/msc/speech_seg/dublin/features/Constant_Quality/',
    'Liverpool':'/home/garrett/Workspace/msc/speech_seg/liverpool/features/Constant_Quality/',
    'London': '/home/garrett/Workspace/msc/speech_seg/london/features/Constant_Quality/',
}

# this is the dataframe that will keep all the feature data
accent_features_df = pd.DataFrame()

# iterate through the utterances and process each one
for utterance in utterances:
    for accent, base_dir in base_directories.items():
        directory = os.path.join(base_dir, utterance, utterance) 
        df = analyse_directory(directory)
        df['Group'] = accent
        df['Utterance'] = utterance
        accent_features_df = pd.concat([accent_features_df, df])

# run all various tests, these include:
# shapiro wilk 
# levene test 
# mann whitney u and permutation test
for utterance in utterances:
    print(" ")
    print(f"statistical test on utterance: '{utterance}'")
    for feature in ['num_peaks', 'mean_energy']:
        print(" ")
        print(f"feature: {feature}")
        #shapiro testing
        for accent in base_directories.keys():
            data = accent_features_df[(accent_features_df['Group'] == accent) & 
                                      (accent_features_df['Utterance'] == utterance)][feature]
            _, p_value_shapiro = stats.shapiro(data)
            print(f"{accent} shapiro wilk p-value on {feature}: {p_value_shapiro}")
        
        #levens testing
        data_groups = [accent_features_df[(accent_features_df['Group'] == accent) & 
                                          (accent_features_df['Utterance'] == utterance)][feature] for accent in base_directories]
        _, p_value_levene = stats.levene(*data_groups)
        print(f"levene test p-val for {feature}: {p_value_levene}")

        # mann whitney testsing
        accents = list(base_directories.keys())
        for i in range(len(accents)-1):
            for x in range(i+1, len(accents)):
                data_1 = accent_features_df[(accent_features_df['Group'] == accents[i]) & 
                                            (accent_features_df['Utterance'] == utterance)][feature]
                data_2 = accent_features_df[(accent_features_df['Group'] == accents[x]) & 
                                            (accent_features_df['Utterance'] == utterance)][feature]
                stat, p_value, p_value_perm = perform_mann_whitney_and_permutation_test(data_1, data_2)
                print(f"mann whitney u and permutation test between accents {accents[i]} and {accents[x]} for {feature}: U={stat}, p-value={p_value}, Permutation p-value={p_value_perm}")
