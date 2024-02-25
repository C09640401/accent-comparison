import librosa
import numpy as np
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu

def calc_mean_pitch(y, sr):
    '''
    get the mean pitch for an audio file containg utterance
    Used the following documentaiton for guidance: 
    pavlos163 (2017)
    '''
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch.append(pitches[index, t])

    pitch = np.array(pitch)
    pitch = pitch[pitch > 0]
    mean_pitch = np.mean(pitch)
    return mean_pitch

def get_pitch(directory):
    mean_pitch_vals = []
    #iterate of the file directory and ge the mean pitch values
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            y, sr = librosa.load(os.path.join(directory, filename))
            pitch = calc_mean_pitch(y, sr)
            mean_pitch_vals.append((filename, pitch))

    return mean_pitch_vals
    
    
def perform_mann_whitney_and_permutation_test(data_1, data_2, perm_count=1000):
    '''
    perform mann whintey permutation testing , this code was guided by (same as pitch):
    
    Cicek (2022)
    and
    Deak (2022)
    
    this function assumes the data is balanced for the 2 accent data sets.
    '''
    original_stat, original_p_value = stats.mannwhitneyu(data_1, data_2, alternative='two-sided')

    combined_data = np.concatenate([data_1, data_2])
    n = len(data_1)

    perm_stats = np.zeros(perm_count)
    for i in range(perm_count):
        np.random.shuffle(combined_data)
        perm_stat, _ = stats.mannwhitneyu(combined_data[:n], combined_data[n:], alternative='two-sided')
        perm_stats[i] = perm_stat

    p_value_perm = np.mean(perm_stats >= original_stat)

    return original_stat, original_p_value, p_value_perm
    

utterances = ['a', 'i', 'o', 'u']

base_directories = {
    'Dublin': '/home/garrett/Workspace/msc/speech_seg/dublin/features/Monophthongs/',
    'London': '/home/garrett/Workspace/msc/speech_seg/london/features/Monophthongs/',
    'Liverpool': '/home/garrett/Workspace/msc/speech_seg/liverpool/features/Monophthongs/'
}

for utterance in utterances:
    pitch_values = {}
    min_count = float('inf')

    for accent, base_dir in base_directories.items():
        directory = os.path.join(base_dir, utterance)
        pitches = get_pitch(directory)
        pitch_values[accent] = pd.Series([pitch for _, pitch in pitches], name=accent)
        min_count = min(min_count, len(pitch_values[accent]))

    for accent in pitch_values:
        print(f"{accent} descriptive stats for '{utterance}':")
        print(pitch_values[accent].describe())
        print(" ")

    plt.figure(figsize=(10, 6))
    for accent, values in pitch_values.items():
        sns.histplot(values, label=accent, kde=True)
    plt.title(f"pitch distribution on '{utterance}'")
    plt.xlabel("pitch in Hz")
    plt.ylabel("freq")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=list(pitch_values.values()), palette=["blue", "red", "yellow"])
    plt.xticks(range(len(pitch_values)), pitch_values.keys())
    plt.title(f"pitch comparison on '{utterance}'")
    plt.ylabel("pitch in Hz")
    plt.show()

    for accent, values in pitch_values.items():
        plt.figure(figsize=(10, 6))
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f"Q-Q plot on {accent} pitch data ('{utterance}')")
        plt.show()

    # shapiro-wilk normality testing
    print(" ")
    print("shapiro-wilk test for normality")
    for accent in pitch_values:
        _, p_value_normality = stats.shapiro(pitch_values[accent])
        print(f"{accent}: p-value = {p_value_normality}")
    print(" ")

    # levene's variance testing
    _, p_value_levene = stats.levene(*pitch_values.values())
    print(" ")
    print("levene's test for homogeneity of variance")
    print(f"p-value = {p_value_levene}")
    print(" ")

    accents = list(pitch_values.keys())
    for i in range(len(accents)):
        for x in range(i + 1, len(accents)):
            perform_mann_whitney_and_permutation_test(pitch_values[accents[i]], pitch_values[accents[x]])
