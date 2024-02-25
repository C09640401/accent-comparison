import os
from pydub import AudioSegment
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import eng_to_ipa as ipa
import time
import scipy.io.wavfile
import shutil

'''
This script creates a new wav file for every phoneme detected in the word files. 
It iterates over the word files that exists in a given base directory (provided via the cofig var).
For each word, extract the phonemes using the Wav2Vec model and save these phonemes as a new wav file.
The work completed in this script was guided by the work seen in :
Xu (2021) and code samples provided in Baevski (2021).
'''


# when rerunning this script, some of residual files were causing issues
# this funciton is used to clean up files for a fresh execution
def clear_directory(directory):

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'error deleting {file_path} : {e}')

def save_phonemes_as_audio(audio, phoneme_list, output_dir, sr):

    audio_segment_data = None
    audio_int16 = None
    start_time_ms= None
    end_time_ms = None
    start_sample = None
    end_sample = None
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    


    for item in phoneme_list:
        phoneme = item['phoneme']
        if phoneme.strip() == '':
            continue

        try:
            start_ms = int(float(item['start_time']))
            end_ms = int(float(item['end_time']))
        except Exception as e:
            print("error casting start and end times: " + str(e))
        
        start_sample = int(start_ms * sr / 1000)
        end_sample = int(end_ms * sr / 1000)

        
        try:
            audio_segment_data = audio[start_sample:end_sample]
        except Exception as e:
            print(f"error creating audio segnment data:  {e}")
            
        # adding 2 second buffer at start and end of audio, this seems to 
        # help in the performance of word and phoneme segmentation
        silence_duration = 2  # seconds
        silence_samples = int(silence_duration * sr)
        silence = np.zeros(silence_samples)
        audio_segment_data_silenced = np.concatenate((silence, audio_segment_data, silence))
        

        filename = os.path.join(output_dir, f"{phoneme}_{start_ms:.2f}_{end_ms:.2f}.wav")
        try:
            scipy.io.wavfile.write(filename, sr, audio_segment_data_silenced)
        except Exception as e:
            print(f"Error exporting audio for phoneme '{phoneme}': {e}")



def process_audio(file_path, word, model, processor, phoneme_directory, chunk_duration=1.0):
    try:
        print(f"Processing file: {file_path}")
        audio = None
        input_values = None
        predicted_ids = None
        transcription = None
        
        audio, orig_sampling_rate = librosa.load(file_path, sr=16000)

        
        audio_silenced = None
        silence_duration = 2
        silence_samples = int(silence_duration * orig_sampling_rate)
        silence = np.zeros(silence_samples)
        audio_silenced = np.concatenate((silence, audio, silence))
        
        total_duration_seconds = len(audio) / orig_sampling_rate

        input_values = processor(audio_silenced, sampling_rate=16000, return_tensors="pt").input_values
        

        with torch.no_grad():
            logits = model(input_values).logits

        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        predicted_ids = torch.argmax(probabilities, dim=-1)
        confidence_scores = probabilities.max(dim=-1).values
        
        transcription = processor.batch_decode(predicted_ids)[0]
        
        print(transcription)
        
        duration_per_token = total_duration_seconds * 1000 / len(transcription)  # milliseconds
        
        phoneme_list = []
        phoneme_confidences = []
        for i, phoneme in enumerate(transcription):
            start_time = i * duration_per_token
            end_time = (i + 1) * duration_per_token
            phoneme_confidence = confidence_scores[0, i].item()  # confidence scor for each phoneme
            phoneme_confidences.append(phoneme_confidence)
            phoneme_list.append({
                'phoneme': phoneme,
                'start_time': f"{start_time:.2f}",
                'end_time': f"{end_time:.2f}",
                'probability': phoneme_confidence
            })
        
        for phoneme in phoneme_list:
            print(f"Phoneme: {phoneme['phoneme']}, Probability: {phoneme['probability']:.4f}")

        
        output_dir = os.path.join(os.path.dirname(file_path), phoneme_directory)
        print(output_dir)
        # clear output directory before processing to avoid duplicates
        if os.path.exists(output_dir):
            clear_directory(output_dir)
            
        save_phonemes_as_audio(audio, phoneme_list, output_dir, orig_sampling_rate)

    except Exception as e:
        print(f"Error processing audio: {e}")
        
    finally:
        pass

def main():

    
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

    try:
        # this is the hugging face model used for phonenem seg
        model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_name)
    except Exception as e:
        print(f"error creating model: {e}")
        
        
        
    try:
        # iterae of the word wav files in each base directory and extract the relevant phonemes
        # from these word files. Each phoneme is saved as a new wav file 
        for accent_config in config:
            base_dir = accent_config["base_destination_dir"]
            
            for word in accent_config["words"]:
                word_dir = os.path.join(base_dir, word)
                if os.path.exists(word_dir):
                    individual_dirs = [d for d in os.listdir(word_dir) if os.path.isdir(os.path.join(word_dir, d))]
                    for individual_dir in individual_dirs:
                        individual_dir_path = os.path.join(word_dir, individual_dir)
                        word_files = [f for f in os.listdir(individual_dir_path) if f.endswith(".wav")]
                        for word_file in word_files:
                            file_path = os.path.join(individual_dir_path, word_file)
                            phoneme_directory = "phonemes_" + str(word_file[:-4])
                            if os.path.exists(file_path):
                                process_audio(file_path, word, model, processor, phoneme_directory)
                                time.sleep(3)
            
            
            
                            
    except Exception as e:
        print(f"error getting words for processing: {e}")

if __name__ == "__main__":
    main()
        
        
        

    

