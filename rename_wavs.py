import os
import pandas as pd

'''
This script is used to rename all of the youtube audio clip file names to numerical values.
Some of the file names were quite long and other contained explicit language. This script 
could also be used to anonymise the file names if needed. 
'''


# get the new names to use from csv sheet 
def load_mapping(file_path):
    df = pd.read_csv(file_path)
    df['new_directory'] = df['new_directory'].astype(int)
    return dict(zip(df['original_name'], df['new_directory']))

# rename the existing directories (which are currently named after the youtube video links)
# to numerical values for some form on anyomonity.
def rename_directories(root_dir, mapping):
    for dir_name in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(subdir_path):
            for sub_dir_name in os.listdir(subdir_path):
                sub_dir_path = os.path.join(subdir_path, sub_dir_name)
                if os.path.isdir(sub_dir_path) and sub_dir_name in mapping:
                    new_name = os.path.join(subdir_path, str(mapping[sub_dir_name]))
                    os.rename(sub_dir_path, new_name)
                    print(f"Renamed '{sub_dir_name}' to '{new_name}'")


file_index_path = '/home/garrett/Workspace/msc/file_index.csv'
#root_directory = '/home/garrett/Workspace/msc/speech_seg/dublin/features/Monophthongs/' 
root_directory = '/home/garrett/Workspace/msc/speech_seg/london/features/words/'
    
mapping = load_mapping(file_index_path)
rename_directories(root_directory, mapping)


