import pandas as pd
import os
import shutil
from tqdm import tqdm
import tensorflow_io as tfio
import tensorflow as tf


# Get hash table
exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
df = pd.read_csv(os.path.join(exec_path, 'raw_data', '_info.csv'))
hash_table = {}
for index, row in df.iterrows():
    file_id = row['FileID']
    
    if row['Negative'] == 1:
        hash_table[file_id] = 0
    if row['Typical'] == 1:
        hash_table[file_id] = 1
    if row['Atypical'] == 1:
        hash_table[file_id] = 2

#%%

# Move train data
for root, dirs, files in os.walk(os.path.join(exec_path, 'raw_data', 'train')):
    for file in files:
        
        if not file.endswith('.dcm'):
            continue
        
        # Get old file's path
        old_file_path = os.path.join(root, file)
        
        # Get new file's path
        file_id = file.replace('.dcm', '')
        file_label = hash_table[file_id]
        new_file_path = os.path.join('__data__', 'train', str(file_label), file.replace('.dcm', '.png'))
        
        # Change to png and store the file to new path
        os.makedirs(os.path.dirname(new_file_path), exist_ok = True)
        image = tfio.image.decode_dicom_image(tf.io.read_file(old_file_path), dtype=tf.uint16)[0]
        tf.keras.utils.save_img(new_file_path, image)

#%%

# Move valid data
count = 1
for root, dirs, files in os.walk(os.path.join(exec_path, 'raw_data', 'valid')):
    for file in files:
        
        if not file.endswith('.dcm'):
            continue
        
        # Get old file's path
        old_file_path = os.path.join(root, file)
        
        # Get new file's path
        new_file_path = os.path.join('__data__', 'test', str(count), file.replace('.dcm', '.png'))
        
        # Change to png and store the file to new path
        os.makedirs(os.path.dirname(new_file_path), exist_ok = True)
        image = tfio.image.decode_dicom_image(tf.io.read_file(old_file_path), dtype=tf.uint16)[0]
        tf.keras.utils.save_img(new_file_path, image)
        count += 1