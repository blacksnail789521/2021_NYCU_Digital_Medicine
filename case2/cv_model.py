# pip install tensorflow_hub
# pip install tensorflow_text
# pip install tf-models-official

#%%

exec('''import sys,os\nexec_path=os.path.abspath(os.path.join(os.path.dirname(__file__),''))\nif exec_path not in sys.path:sys.path.insert(0,exec_path)''')
import os
import shutil
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tf's warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%

data_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '__data__'))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
os.makedirs(output_path, exist_ok = True)

#%%

def setup_GPU(max_gpu = 1):
        
    # Determine starting_gpu
    if len(sys.argv) == 1:
        starting_gpu = 3
    else:
        starting_gpu = int(sys.argv[1])
    
    # Set GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[ starting_gpu : starting_gpu + max_gpu ], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    
    
    # Print GPU usage
    print('------------------------------')
    print(physical_devices[ starting_gpu : starting_gpu + max_gpu ])
    print('len of GPU:', len(logical_devices))
    
    assert len(logical_devices) <= max_gpu

setup_GPU()

#%%

def read_data_as_ds(batch_size = 16, validation_split = 0.2, seed = None, color_mode = 'grayscale', peek = False):
    
    def peek_ds(ds, class_names):
        
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
    
    
    ''' Main function of read_data_as_ds '''
    
    assert color_mode in ['grayscale', 'rgb']
    
    if seed is None:
        seed = np.random.randint(1e6)
    
    if validation_split == 0:
        # Get train_ds
        train_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(data_root_path, 'train'),
            batch_size = batch_size,
            seed = seed,
            color_mode = color_mode)
        class_names = train_ds.class_names
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        valid_ds = None
    else:
    
        # Get train_ds
        train_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(data_root_path, 'train'),
            batch_size = batch_size,
            validation_split = validation_split,
            subset = 'training',
            seed = seed,
            color_mode = color_mode)
        class_names = train_ds.class_names
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Get valid_ds
        valid_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(data_root_path, 'train'),
            batch_size = batch_size,
            validation_split = validation_split,
            subset = 'validation',
            seed = seed,
            color_mode = color_mode)
        valid_ds = valid_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Peek data if we need
    if peek:
        peek_ds(train_ds, class_names)
    
    return train_ds, valid_ds

# validation_split = 0 # Don't split
validation_split = 0.2

# color_mode = 'grayscale'
color_mode = 'rgb'

peek = False


train_ds, valid_ds = read_data_as_ds(validation_split = validation_split, \
                                     color_mode = color_mode)

IMG_SIZE = (256, 256)

x, y = next(iter(train_ds))
y = y.numpy()

#%%

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

show_augmeted = False
# show_augmeted = True

if show_augmeted:
    for images, _ in train_ds.take(1):
        
        image = images[0]
        
        # Show original image
        plt.imshow(image.numpy().astype("uint8"))
        plt.axis("off")
        plt.show()
                    
        # Show augmented images
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')


#%%

def get_efficient_model(weights = 'imagenet'):
    
    preprocessing_model = tf.keras.applications.efficientnet.preprocess_input
    
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape = (256, 256, 3),
        include_top = False,
        weights = weights
    )
    
    return preprocessing_model, base_model
    
weights = 'imagenet'
# weights = None

preprocessing_model, base_model = get_efficient_model(weights = weights)


#%%

def get_final_model(preprocessing_model, base_model, train_base_model = True):
    
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = data_augmentation(inputs)
    x = preprocessing_model(x)
    x = base_model(x, training = train_base_model)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(3)(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model


train_base_model = True
# train_base_model = False

model = get_final_model(preprocessing_model, base_model, train_base_model)

model.summary()

#%%

def compile_model(model):
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model

model = compile_model(model)

#%%

def train_model(model, train_ds, valid_ds, epochs = 10, plot = True):
    
    history = model.fit(train_ds,
                        epochs = epochs,
                        validation_data = valid_ds)
    
    # Plot learning curve
    if plot:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        # plt.show()
        plt.savefig('learning_curve.png')

train_model(model, train_ds, valid_ds, epochs = 20)

#%%

def output_csv(model, color_mode):
    
    output_dict = {'FileID': [], 'Type': []}
    for file_index in os.listdir(os.path.join(data_root_path, 'test')):
        file_folder = os.path.join(data_root_path, 'test', file_index)
        file_name = os.listdir(file_folder)[0]
        file_id = file_name.replace('.png', '')
        # Get test_image
        test_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(data_root_path, 'test', file_index),
            batch_size = 1,
            color_mode = color_mode,
            labels = None)
        test_image = next(iter(test_ds)).numpy()
        
        # Get prediction
        y_pred = model(test_image).numpy()
        score = tf.nn.softmax(y_pred[0])
        print(np.argmax(score))
        
        if np.argmax(score) == 0:
            file_label = 'Negative'
        elif np.argmax(score) == 1:
            file_label = 'Typical'
        elif np.argmax(score) == 2:
            file_label = 'Atypical'
        
        output_dict['FileID'].append(file_id)
        output_dict['Type'].append(file_label)
    
    df = pd.DataFrame(output_dict)
    df.to_csv('output.csv', index = False)
    
    return df


result = output_csv(model, color_mode)