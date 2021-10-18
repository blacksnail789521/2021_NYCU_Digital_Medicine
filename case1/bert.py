# pip install tensorflow_hub
# pip install tensorflow_text
# pip install tf-models-official

#%%

exec("""import sys,os\nexec_path=os.path.abspath(os.path.join(os.path.dirname(__file__),""))\nif exec_path not in sys.path:sys.path.insert(0,exec_path)""")
import os
import shutil
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Disable tf"s warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt

from bert_model_list import bert_model_list

#%%

data_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Data'))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
os.makedirs(output_path, exist_ok=True)

#%%

def setup_GPU(max_gpu = 1):
        
    # Determine starting_gpu
    if len(sys.argv) == 1:
        starting_gpu = 3
    else:
        starting_gpu = int(sys.argv[1])
    
    # Set GPU
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[ starting_gpu : starting_gpu + max_gpu ], "GPU")
    logical_devices = tf.config.list_logical_devices("GPU")
    
    
    # Print GPU usage
    print("------------------------------")
    print(physical_devices[ starting_gpu : starting_gpu + max_gpu ])
    print("len of GPU:", len(logical_devices))
    
    assert len(logical_devices) <= max_gpu

setup_GPU()

#%%

def make_data_tf_data_friendly():
    
    for data_type in ['train', 'test']:
        if data_type == 'train':
            raw_path = os.path.join(data_root_path, 'Train_Textual')
        elif data_type == 'test':
            raw_path = os.path.join(data_root_path, 'Test_Intuitive')
        
        # Create folders for pos and neg
        new_path = os.path.join(data_root_path, data_type)
        os.makedirs(os.path.join(new_path, 'pos'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'neg'), exist_ok=True)
        for file_name in os.listdir(raw_path):
            if file_name.startswith('Y'):
                # pos
                shutil.copy(os.path.join(raw_path, file_name), os.path.join(new_path, 'pos', file_name))
            else:
                # neg
                shutil.copy(os.path.join(raw_path, file_name), os.path.join(new_path, 'neg', file_name))

make_data_tf_data_friendly()

#%%

def read_data_as_ds(batch_size = 16, validation_split = 0.2, seed = None, peek = False):
    
    def peek_ds(ds, class_names):
        
        # Try to observe one batch
        for text_batch, label_batch in train_ds.take(1):
            # We only care for the first three examples
            for i in range(3):
                print('--------------------------------------------------')
                print(f'Review: {text_batch.numpy()[i]}')
                label = label_batch.numpy()[i]
                print(f'Label : {label} ({class_names[label]})')
    
    
    ''' Main function of read_data_as_ds '''
    
    if seed is None:
        seed = np.random.randint(1e6)
    
    if validation_split == 0:
        # Get train_ds
        train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            os.path.join(data_root_path, 'train'),
            batch_size=batch_size,
            seed=seed)
        class_names = train_ds.class_names
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        val_ds = None
    else:
    
        # Get train_ds
        train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            os.path.join(data_root_path, 'train'),
            batch_size=batch_size,
            validation_split=validation_split,
            subset='training',
            seed=seed)
        class_names = train_ds.class_names
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Get val_ds
        val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            os.path.join(data_root_path, 'train'),
            batch_size=batch_size,
            validation_split=validation_split,
            subset='validation',
            seed=seed)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Get test_ds
    test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        os.path.join(data_root_path, 'test'),
        batch_size=batch_size)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Peek data if we need
    if peek:
        peek_ds(train_ds, class_names)
    
    return train_ds, val_ds, test_ds

# validation_split = 0 # Don't split
validation_split = 0.2
train_ds, val_ds, test_ds = read_data_as_ds(validation_split=validation_split)

#%%

def get_bert_models(model_type = 'small_bert', print_url = False):
    
    if model_type == 'small_bert':
        bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
    elif model_type == 'bert_experts':
        bert_model_name = 'experts_pubmed'
    elif model_type == 'electra':
        bert_model_name = 'electra_base'
    elif model_type == 'albert':
        bert_model_name = 'albert_en_base'
    tfhub_handle_encoder, tfhub_handle_preprocess = bert_model_list(bert_model_name)
    
    if print_url:
        print(f'BERT model selected           : {tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    bert_model = hub.KerasLayer(tfhub_handle_encoder, trainable = True, name='BERT_encoder')
    
    return bert_preprocess_model, bert_model

model_type = 'small_bert'
# model_type = 'bert_experts'
# model_type = 'electra'
# model_type = 'albert'
bert_preprocess_model, bert_model = get_bert_models(model_type)

#%%

def build_model(bert_preprocess_model, bert_model, plot_model = True):
    
    # A very simple fine-tuned model, with the preprocessing model, 
    # the selected BERT model, one Dense and a Dropout layer.
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    
    preprocessed_inputs = bert_preprocess_model(inputs)
    
    bert_outputs = bert_model(preprocessed_inputs)
    outputs = bert_outputs['pooled_output']
    
    # outputs = tf.keras.layers.Dropout(0.3)(outputs)
    # outputs = tf.keras.layers.Dense(100, activation='relu', \
    #                                 kernel_regularizer=tf.keras.regularizers.L2(0.1), \
    #                                 name='dense_1')(outputs)
    outputs = tf.keras.layers.Dropout(0.5)(outputs)
    outputs = tf.keras.layers.Dense(1, activation=None, \
                                    kernel_regularizer=tf.keras.regularizers.L2(0.01), \
                                    name='dense_output')(outputs)
    
    model = tf.keras.Model(inputs, outputs)
    
    if plot_model:
        tf.keras.utils.plot_model(model, to_file = os.path.join(output_path, 'model.png'))
    
    return model

model = build_model(bert_preprocess_model, bert_model)

#%%

def train_model(model, train_ds, val_ds, epochs = 30):

    # Compile
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.metrics.BinaryAccuracy(), tfa.metrics.F1Score(num_classes=1, threshold=0.5)]

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    
    init_lr = 3e-5
    # init_lr = 1e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Fit
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta = 1e-5, patience=10, verbose=1)
    history = model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=epochs,
        # callbacks=[early_stopping],
    )
    
    
    return model, history

model, history = train_model(model, train_ds, val_ds)

#%%

def plot_training_results(history):
    
    history_dict = history.history
    # print(history_dict.keys())
    
    epochs = range(1, len(history_dict['loss']) + 1)
    fig = plt.figure(figsize=(10, 6))
    
    # loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, history_dict['loss'], 'r', label='Training loss')
    if history_dict.get('val_loss', None) is not None:
        plt.plot(epochs, history_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.xticks(epochs)
    
    # accuracy
    plt.subplot(3, 1, 2)
    plt.plot(epochs, history_dict['binary_accuracy'], 'r', label='Training acc')
    if history_dict.get('val_loss', None) is not None:
        plt.plot(epochs, history_dict['val_binary_accuracy'], 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.xticks(epochs)
    
    # f1_score
    plt.subplot(3, 1, 3)
    plt.plot(epochs, history_dict['f1_score'], 'r', label='Training f1_score')
    if history_dict.get('val_loss', None) is not None:
        plt.plot(epochs, history_dict['val_f1_score'], 'b', label='Validation f1_score')
    plt.title('Training and validation f1_score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    # plt.xticks(epochs)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, f'{model_type}.png'))

plot_training_results(history)

#%%

def evaluate_model(model, test_ds):
    # This is just for observation purposes.
    loss, accuracy, f1_score = model.evaluate(test_ds)
    
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    print(f'F1 score: {f1_score[0]}')

evaluate_model(model, test_ds)

#%%

def predict_with_model(model):
    
    # Get data for prediction
    predict_data_list = []
    predict_filename_list = list(sorted(os.listdir(os.path.join(data_root_path, 'Validation'))))
    for file_name in predict_filename_list:
        with open(os.path.join(data_root_path, 'Validation', file_name), 'r') as f:
            predict_data = f.read()
            predict_data_list.append(predict_data)
    
    # Use model to predict
    predict = model.predict(predict_data_list)
    predict = tf.sigmoid(predict).numpy().flatten()
    
    # Output results
    output_df = pd.DataFrame()
    output_df['Filename'] = predict_filename_list
    output_df['Obesity'] = [ 1 if p >= 0.5 else 0 for p in predict ]
    
    output_df.to_csv(os.path.join(output_path, 'output.csv'), index=False)

predict_with_model(model)