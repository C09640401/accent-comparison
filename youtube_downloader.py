import os
from pytube import YouTube

def download_youtube_audio(url, output_dir='youtube_downloads'):
    '''
    This function is for downloading audio content from youtube video clips.
    These are saved as wav files in the local directory. This code was guided by
    work completed in Ficano. (2023)
    '''
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # using youtube library to instantiate youtube instance
        yt = YouTube(url)

        audio_stream = yt.streams.filter(only_audio=True, 
                                               file_extension='webm').first()

        # save the best quality audio_stream to local storage
        audio_stream.download(output_path=output_dir)

        # this project uses wav file extensions, ensure this is the extension used
        original_file_path = os.path.join(output_dir, audio_stream.default_filename)
        wav_file_path = os.path.join(output_dir, f"{you_tube.title}.wav")
        os.rename(original_file_path, wav_file_path)

    except Exception as e:
        print(f"something went wrong: {e}")


youtube_url = input("YouTube URL to download: ")
download_youtube_audio(youtube_url)
