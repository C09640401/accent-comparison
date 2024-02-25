import os
import librosa
import numpy as np
import pandas as pd
import random
import scipy.stats as stats
# if you are not running this script in a notebook environment, you may need to import
# perform_mann_whitney_and_permutation_test from the pitch script.

def calculate_mean_intensity(file_path):
    '''
    this function is responsible for calculating the average loudness of the passed in file path.
    the average loudness is commonly referred to as the audio intensity. Guidance for this function 
    was taken from:
    Harris et al. (2020)
    and 
    AJonas Adler (2022)
    '''
    y, sr = librosa.load(file_path, sr=None)
    absolute_intensity_val  = np.abs(y) #not sure if np (absolute or abs) function should be used here
    mean_intensity = np.mean(absolute_intensity_val)
    return mean_intensity

def process_directory(directory):
    intensities = []
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            file_path = os.path.join(directory, file)
            intensity = calculate_mean_intensity(file_path)
            intensities.append((file, intensity))
    return intensities


for utterance in utterances:
    print(f"running for utterance: '{utterance}'")
    all_data = {}

    for accent, base_dir in base_directories.items():
        directory = os.path.join(base_dir, utterance)
        intensities = process_directory(directory)
        df = pd.DataFrame(intensities, columns=['filename', 'intensity'])
        all_data[accent] = df

    # balancing the data between the accents, have some discrepency in some cases
    min_count = min(len(all_data[accent]) for accent in all_data)
    for accent in all_data:
        if len(all_data[accent]) > min_count:
            all_data[accent] = all_data[accent].sample(n=min_count, random_state=8)
    
    # mann whitney U test resutls and permutation results for each set of accents
    print("mann whitney U and permutation results for utterance:", utterance)
    accents = list(all_data.keys())
    for i in range(len(accents)):
        for j in range(i+1, len(accents)):
            accent_1, accent_2 = accents[i], accents[j]
            data_1 = all_data[accent_1]['intensity']
            data_2 = all_data[accent_2]['intensity']
            stat, p_value, p_value_perm = perform_mann_whitney_and_permutation_test(data_1.values, data_2.values)
            print(f"between {accent_1} and {accent_2}: U={stat}, p-value={p_value}, permutation p-value={p_value_perm}")
    print(" ")
