import librosa
import numpy as np
import pandas as pd
import os
import random
import scipy.stats as stats
# if you are not running this script in a notebook environment, you may need to import
# perform_mann_whitney_and_permutation_test from the pitch script.

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def trim_silence(audio, sr):
    '''
    A silence buffer was applied to improve the performance the NN model.
    This is a quick access function added to trim the buffer off when needed. 
    '''
    return audio[2*sr:-2*sr]

def get_formants(audio, sr):
    '''
    this code detects the top 3 freqency peaks which represents the formants.
    This code was guided by the work completed in:
    Rathee, A. (2020)
    and 
    rayryeng. (2020)
    and
    McFee et al. (2015) 
    '''
    magnitude_spec = np.abs(librosa.stft(audio))
    mean_spectrum = np.mean(magnitude_spec, axis=1)
    peaks = librosa.util.peak_pick(mean_spectrum, pre_max=3, 
                                   post_max=3, pre_avg=3, 
                                   post_avg=5, delta=0.5, wait=10)
    formants = sorted(peaks[:3])
    # convert formant frequencies (top 3 obtained in previous line of code, 
    # to the matching frequency values in hertz which is what is needed for the analysis
    formants_hz = librosa.core.fft_frequencies(sr=sr)[formants]
    return formants_hz if len(formants_hz) == 3 else [0, 0, 0]


def process_directory(directory):
    formant_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            audio, sr = load_audio(os.path.join(directory, filename))
            audio = trim_silence(audio, sr)
            formants = get_formants(audio, sr)
            formant_data.append([filename] + list(formants))
    return pd.DataFrame(formant_data, columns=['Filename', 'F1', 'F2', 'F3'])


# iterate through the utterances
for utterance in utterances:
    print(f"utternance: '{utterance}'")
    accent_data = {}


    for accent, base_dir in base_directories.items():
        directory = os.path.join(base_dir, utterance)
        data = process_directory(directory)
        accent_data[accent] = data

    # balance the data sets, this is only needed if the data is imbalanced, which it was for this proejct
    min_count = min(len(accent_data[accent]) for accent in accent_data)
    for accent in accent_data:
        if len(accent_data[accent]) > min_count:
            accent_data[accent] = accent_data[accent].sample(min_count, random_state=8)

        print(f"{accent} descriptive stats for utterance '{utterance}':")
        print(accent_data[accent].describe().drop('count'))
        print(" ")

    # get the mann whitney U test results and execute and 
    # execute a permutation test for each formant also
    print("mann whitney U and perm test results")
    for formant in ['F1', 'F2', 'F3']:
        for accent_1 in base_directories:
            for accent_2 in base_directories:
                if accent_1 < accent_2:  # there was an issue with duplication on the results
                    formant_data_1 = accent_data[accent_1][formant]
                    formant_data_2 = accent_data[accent_2][formant]
                    stat, p_value, p_value_perm = perform_mann_whitney_and_permutation_test(formant_data_1, 
                                                                                            formant_data_2)
                    # for each formant, print out the results so these can be copied into a table for report
                    print(f"{formant} between {accent_1} and {accent_2} for '{utterance}': U={stat}, p-value={p_value}, Permutation p-value={p_value_perm}")
    print(" ")
