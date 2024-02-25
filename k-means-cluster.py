import librosa
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score
import seaborn as sns

def mean_pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        if magnitudes[index, t] > 0:
            pitch.append(pitches[index, t])

    if len(pitch) == 0:
        return 0 

    return np.mean(pitch)

def detect_sound_duration(file_path):
    y, sr = librosa.load(file_path, sr=None)
    hop_length = 512
    frame_length = 1024
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])
    
    normalised_energy = energy / np.max(energy)
    threshold = 0.01
    onset_frames = np.where(normalised_energy > threshold)[0]
    
    if len(onset_frames) == 0:
        return 0
    
    onset_time = librosa.frames_to_time(onset_frames[0], sr=sr, hop_length=hop_length)
    offset_time = librosa.frames_to_time(onset_frames[-1], sr=sr, hop_length=hop_length)
    
    return offset_time - onset_time

def calculate_mean_intensity(y):
    return np.mean(np.abs(y))

def trim_silence(audio, sr):
    return audio[2*sr:-2*sr] 


def analyse_directory(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            y, sr = librosa.load(file_path)
            audio_trimmed = trim_silence(y, sr)
            pitch = mean_pitch(y, sr)
            duration = detect_sound_duration(file_path)
            intensity = calculate_mean_intensity(y)
            formants = get_formants(audio_trimmed, sr)
            data.append([filename, pitch, duration, intensity] + list(formants))
    
    columns = ['Filename', 'Pitch', 'Duration', 'Intensity', 'F1', 'F2', 'F3']
    df = pd.DataFrame(data, columns=columns)
    return df

# back to using vowel utterances
vowels = ['a', 'i', 'o', 'u']
base_directories = {
    'Dublin': '/home/garrett/Workspace/msc/speech_seg/dublin/features/Monophthongs/',
    'Liverpool': '/home/garrett/Workspace/msc/speech_seg/liverpool/features/Monophthongs/',
}
group_to_label = {'Dublin': 0, 'Liverpool': 1}


for vowel in vowels:
    combined_df = pd.DataFrame()
    for city, base_dir in base_directories.items():
        directory = os.path.join(base_dir, vowel)
        df = analyse_directory(directory)
        df['Group'] = city
        df['Vowel'] = vowel
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # balance the dataset since there is some imbalancing between the 2 groups
    min_count = combined_df['Group'].value_counts().min()
    balanced_df = pd.DataFrame()
    for city in base_directories.keys():
        city_df = combined_df[combined_df['Group'] == city]
        balanced_df = pd.concat([balanced_df, city_df.sample(n=min_count, random_state=42)], ignore_index=True)
    
    balanced_df['TrueLabel'] = balanced_df['Group'].map(group_to_label)
    
    # features to use for the clustering model
    features = ['Pitch', 'Duration', 'Intensity', 'F1']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(balanced_df[features])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(scaled_features)
    balanced_df['Cluster'] = kmeans.labels_
    
    # PCA for visualization - There is no real benefit to using this other than educational
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_features)
    plt.figure(figsize=(10, 6))
    for i in range(kmeans.n_clusters):
        plt.scatter(principalComponents[balanced_df['Cluster'] == i, 0], principalComponents[balanced_df['Cluster'] == i, 1], label=f'Cluster {i}')
    plt.legend()
    plt.title(f'PCA Cluster Visualization for Vowel: {vowel}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    
    # create conf matrix and include the scores in outout
    cm = confusion_matrix(balanced_df['TrueLabel'], balanced_df['Cluster'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=group_to_label.keys(), yticklabels=group_to_label.keys())
    plt.title(f'confusion matrix on vowel: {vowel}')
    plt.ylabel('true label')
    plt.xlabel('cluster label')
    plt.show()
    
    # show performance scoring
    accuracy = accuracy_score(balanced_df['TrueLabel'], balanced_df['Cluster'])
    precision = precision_score(balanced_df['TrueLabel'], balanced_df['Cluster'], zero_division=0)
    recall = recall_score(balanced_df['TrueLabel'], balanced_df['Cluster'], zero_division=0)
    f1 = f1_score(balanced_df['TrueLabel'], balanced_df['Cluster'], zero_division=0)
    ari = adjusted_rand_score(balanced_df['TrueLabel'], balanced_df['Cluster'])
    print(f"vowel: {vowel}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1 score: {f1}, adjusted rand index: {ari}")
    print("-" * 50)
