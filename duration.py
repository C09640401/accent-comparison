import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from scipy import stats
import numpy as np
# if you are not running this script in a notebook environment, you may need to import
# perform_mann_whitney_and_permutation_test from the pitch script.

def calculate_durations(directory, accent, sound):
    durations = []
    print("DIRECTORY: " + str(directory))
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            print("FILENAME: " + str(filename))
            audio_path = os.path.join(directory, filename)
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(audio, sr=sr) - 4
            durations.append({'accent': accent, 'duration': duration, 'dound': sound})
            print(durations)
    return durations



def analyse_durations(dataframe, sounds, base_directories):
    for utterance in utterances:
        current_utterance_data = dataframe[dataframe['utterance'] == utterance]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=current_utterance_data, x='duration', hue='accent', element='step', 
                     stat='density', common_norm=False)
        plt.title(f"distribution of '{utterance}' sound durations")
        plt.xlabel('duration (seconds)')
        plt.ylabel('density')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=current_utterance_data, x='accent', y='duration', palette=["blue", "red", "yellow"])
        plt.title(f"boxplot of '{utterance}' utterance durations")
        plt.xlabel('accent')
        plt.ylabel('duration in seconds')
        plt.show()

        accents = list(base_directories.keys())
        for i in range(len(accents)):
            for x in range(i+1, len(accents)):
                data_1 = current_sound_data[current_sound_data['accent'] == accents[i]]['duration'].to_numpy()
                data_2 = current_sound_data[current_sound_data['accent'] == accents[x]]['duration'].to_numpy()
                original_stat, original_p_value, p_value_perm = perform_mann_whitney_and_permutation_test(data_1, 
                                                                                                          data_2)
                print(f"mann whitney U test between {accents[i]} and {accents[x]} for utterance '{utterance}': U={original_stat}, Original p-value={original_p_value}, Permutation p-value={p_value_perm}")

duration_data = pd.DataFrame()

for utterance in utterances:
    for accent, base_dir in base_directories.items():
        directory = os.path.join(base_dir, utterance)
        duration_data = pd.concat([duration_data, 
                                       pd.DataFrame(calculate_durations(directory, accent, utterance))])

analyse_durations(duration_data, utterances, base_directories)
