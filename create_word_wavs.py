import whisperx
from pydub import AudioSegment
import librosa
import numpy as np
import os

'''
This script creates new wav files containing single word utteranes. These words are passed in via a configuration 
object that contains a list of targets words to extract from the full youtube audio clips.
This sript also creates a corresponding text file with the transcription and the start and end times for each word in the file.
The implmentation of this script was guided strongly by the work completed in:
Bain (2023) which includes easy to follow instructions on getting WhisperX up and running
'''


def preprocess_audio(audio_path, desired_sample_rate=16000):

    audio = AudioSegment.from_file(audio_path)

    # resample to match the wav2vec frame rate
    audio = audio.set_frame_rate(desired_sample_rate)
    return audio

def transcribe_and_save_timestamps(audio_path, output_path, model, device, batch_size):
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, language="en", batch_size=batch_size)

    whisperx_model, metadata = whisperx.load_align_model(language_code=result["language"], 
                                                  device=device)
    aligned_result = whisperx.align(result["segments"], 
                                    whisperx_model, 
                                    metadata, 
                                    audio, 
                                    device=device)

    with open(output_path, 'w') as trans_file:
        for segment in aligned_result["segments"]:
            for words in segment["words"]:
                # in some cases, there is no start and end time, need to handle this through some if statements
                if 'start' in words and 'end' in words and 'score' in words:
                    trans_file.write(f"{words['word']} {words['start']} {words['end']} {words['score']}\n")
                else:
                    trans_file.write(f"{words['word']} - - -\n")  # hitting some missing data e.g. years 2020


def get_top_3_timestamps_from_file(timestamp_file, target_word):
    scores = []

    with open(timestamp_file, 'r') as file:
        for line in file:
            word, start, end, score = line.strip().split()
            if word.lower() == target_word.lower():
                scores.append((float(start), float(end), float(score)))

    # sort the list by probabilty score in descending order and return the top 3
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:3]


config = [
    {
        "desired_accent": "Irish English",
        "feature": "words",
        "base_destination_dir": "/home/garrett/Workspace/msc/speech_seg/london/features/words/",
        "words": ["beat", "feet", "seat", "meet", "greet", "heat", "neat", "sweet", "treat", "complete", "bit", "sit",
                             "kit", "lit", "fit", "hit", "tip", "man", "pan", "can", "fan", "car", "ban", "ran", "van", "plan", "hand", "land",
                             "not", "stand", "boat", "coat", "float", "vote", "note", "throat", "wrote", "quote", "gloat", "roam",
                             "but", "book", "word", "hate", "bite", "light", "might", "fight", "night", "right", "sight",
                             "tight", "write", "bright", "height", "delight", "about", "shout", "doubt", "scout", "proud", "loud",
                             "crowd", "mouth", "south", "boot", "shoot", "fruit", "group", "suit", "mute", "root",
                             "thought", "caught", "taught", "bought", "fought", "road", "load", "mode", "remote", "hope", 
                 "rope", "slope", "loaf", "goat", "host", "most", "roast", "toast"]
  
        
    },

    ]



# these are not typical asr settings, had to pass these in to overcome a bug in whisperx
# the model would not let me continue uness temperature setting was passed in 
# please see bug: https://github.com/SYSTRAN/faster-whisper/issues/455
asr_options = {
    "repetition_penalty": 1, 
    "prompt_reset_on_temperature": 0.5,
    "no_repeat_ngram_size": 0
}

device = "cpu"
batch_size = 8
model = whisperx.load_model("small.en", device, compute_type="float32", asr_options=asr_options)

audio_data_dir = "/home/garrett/Workspace/msc/speech_seg/london/audio_data/"
features_dir = "/home/garrett/Workspace/msc/speech_seg/london/features/"
audio_data_cleaned_dir = "/home/garrett/Workspace/msc/speech_seg/london/audio_data_cleaned/"

# transcribe audio file just once and save timestamps, if transcribed file already exists, use it
for filename in os.listdir(audio_data_dir):
    if filename.endswith('.wav'):
        audio_path = os.path.join(audio_data_dir, filename)
        destination_wav_path = os.path.join(audio_data_cleaned_dir, filename)
        timestamp_file = os.path.join(audio_data_cleaned_dir, f"{filename}.txt")
        
        if not os.path.exists(timestamp_file):
            print(f"Transcribing AUDIO: {audio_path}")
            audio = preprocess_audio(audio_path)
            audio.export(destination_wav_path, format='wav')
            transcribe_and_save_timestamps(destination_wav_path, timestamp_file, model, device, batch_size)
        else:
            print(f"Timestamps already exist for {filename}, skipping transcription.")
            
            

# iterate over the features
for feature_config in config:
    feature = feature_config["feature"]
    words = feature_config["words"]

    for filename in os.listdir(audio_data_dir):
        if filename.endswith('.wav'):
            print(f"processing {feature} for audiofile: {filename}")
            timestamp_file = os.path.join(audio_data_cleaned_dir, f"{filename}.txt")

            for word in words:
                top_timestamps = get_top_3_timestamps_from_file(timestamp_file, word)
                for index, (start_sec, end_sec, score) in enumerate(top_timestamps):
                    start_ms = int(start_sec * 1000)
                    end_ms = int(end_sec * 1000)
                    
                    audio = AudioSegment.from_file(os.path.join(audio_data_cleaned_dir, filename))
                    word_audio = audio[start_ms:end_ms]

                    save_dir = os.path.join(features_dir, feature, word)
                    os.makedirs(save_dir, exist_ok=True)

                    file_name_without_extension = filename[:-4]
                    file_dir = os.path.join(save_dir, file_name_without_extension)
                    os.makedirs(file_dir, exist_ok=True)  

                    save_path = os.path.join(file_dir, f"{word}_{index + 1}.wav") #index to filename
                    word_audio.export(save_path, format='wav')
