"""
Reference: MD.ai, Inc.
*Copyright 2018-2020 MD.ai, Inc.   
Licensed under the Apache License, Version 2.0*
"""
import os
import random
import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras_unet.utils import get_augmented
import matplotlib.pyplot as plt
from keras_unet.models import custom_unet
import tensorflow as tf
from keras_unet.metrics import iou
from pydicom import dcmread
import torchvision.transforms as transforms
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader

import mdai
mdai.__version__

def seg_preprocessing(img):
    img = (img / img.max()) * 255
    t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(128), 
                transforms.CenterCrop(128),])

    img = np.array(img, dtype=np.float32)
    img = np.array(img)
    img = t(img)
    img = np.array(img)
    return img
    # img = img.astype(np.uint8)
    # clahe = cv2.createCLAHE(clipLimit = 20.0, tileGridSize = (8,8))
    # return clahe.apply(img)

def train_segmentation_model():
    mdai_client = mdai.Client(domain='public.md.ai', access_token="8a6a548a5485c5e0254ed01099a71c68")
    p = mdai_client.project('aGq4k6NW', path='./lesson2-data')
    p.show_label_groups()

    # this maps label ids to class ids as a dict obj
    labels_dict = {
        'L_A8Jm3d': 1, # Lung   
    }

    print(labels_dict)
    p.set_labels_dict(labels_dict)

    p.show_datasets()
    dataset = p.get_dataset_by_id('D_rQLwzo')
    dataset.prepare()
    image_ids = dataset.get_image_ids()

    # visualize a few train images 
    mdai.visualize.display_images(image_ids[:3], cols=2)

    imgs_anns_dict = dataset.imgs_anns_dict

    def load_images(imgs_anns_dict, img_size=128):
        images = []
        masks = []

        for img_fp in imgs_anns_dict.keys():
            img = mdai.visualize.load_dicom_image(img_fp)
            ann = imgs_anns_dict[img_fp]

            img_width = img.shape[1]
            img_height = img.shape[0]

            mask = np.zeros((img_height, img_width), dtype=np.uint8)

            assert img.shape == mask.shape

            for a in ann:
                vertices = np.array(a['data']['vertices'])
                vertices = vertices.reshape((-1, 2))
                cv2.fillPoly(mask, np.int32([vertices]), (255, 255, 255))

            # resizing and padding
            if img.shape[0] == img.shape[1]:
                resized_shape = (img_size, img_size)
                offset = (0, 0)

            # height > width
            elif img.shape[0] > img.shape[1]:
                resized_shape = (img_size, round(img_size * img.shape[1] / img.shape[0]))
                offset = (0, (img_size - resized_shape[1]) // 2)

            else:
                resized_shape = (round(img_size * img.shape[0] / img.shape[1]), img_size)
                offset = ((img_size - resized_shape[0]) // 2, 0)

            resized_shape = (resized_shape[1], resized_shape[0])
            img_resized = cv2.resize(img, resized_shape).astype(np.uint8)
            mask_resized = cv2.resize(mask, resized_shape).astype(np.bool)

            resized_shape = (resized_shape[1], resized_shape[0])

            # add padding to square
            img_padded = np.zeros((img_size, img_size), dtype=np.uint8)
            img_padded[
                offset[0] : (offset[0] + resized_shape[0]), offset[1] : (offset[1] + resized_shape[1])
            ] = img_resized
            mask_padded = np.zeros((img_size, img_size), dtype=np.bool)
            mask_padded[
                offset[0] : (offset[0] + resized_shape[0]), offset[1] : (offset[1] + resized_shape[1])
            ] = mask_resized

            images.append(img_padded)
            masks.append(mask_padded)

        # add channel dim
        images = np.asarray(images)[:, :, :, None]
        masks = np.asarray(masks)[:, :, :, None]
        return images, masks

    images, masks = load_images(imgs_anns_dict)
    img_index = random.choice(range(len(imgs_anns_dict)))

    print(img_index)
    img_fps = list(imgs_anns_dict.keys())
    img_fp = img_fps[img_index]
    img = mdai.visualize.load_dicom_image(img_fp)
    ann = imgs_anns_dict[img_fp]
    img_width = img.shape[1]
    img_height = img.shape[0]

    mask = np.zeros((img_height, img_width), dtype=np.uint8) 
    for a in ann:     
        vertices = np.array(a['data']['vertices'])
        vertices = vertices.reshape((-1,2))                     
        cv2.fillPoly(mask, np.int32([vertices]), (255,255,255))
        
    plt.figure(figsize=(20, 10))
    plt.subplot(2,3,1)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(mask, cmap=plt.cm.bone)
    plt.axis('off')

    plt.subplot(2,3,3)              
    plt.imshow(cv2.bitwise_and(img, img, mask=mask.astype(np.uint8)), cmap=plt.cm.bone)
    plt.axis('off')

    plt.subplot(2,3,4)
    plt.imshow(images[img_index,:,:,0], cmap=plt.cm.bone)
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.imshow(masks[img_index,:,:,0], cmap=plt.cm.bone)
    plt.axis('off')

    plt.subplot(2,3,6)
    plt.imshow(cv2.bitwise_and(images[img_index,:,:,0], images[img_index,:,:,0], 
                            mask=masks[img_index,:,:,0].astype(np.uint8)), cmap=plt.cm.bone)
    plt.axis('off')


    x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=0)

    train_gen = get_augmented(
        x_train, y_train, batch_size=8,
        data_gen_args = dict(
            rotation_range=180,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
        ))

    seg_model = custom_unet(
        x_train[0].shape,
        use_batch_norm=True,
        num_classes=1,
        filters=64,
        dropout=0.2,
        output_activation='sigmoid',
    )

    """### Train model"""

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
        ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ]

    seg_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy', iou],
    )

    history = seg_model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=10,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    print(history.history.keys())

    plt.figure()
    plt.plot(history.history['accuracy'], 'orange', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'blue', label='Validation accuracy')
    plt.plot(history.history['loss'], 'red', label='Training loss')
    plt.plot(history.history['val_loss'], 'green', label='Validation loss')
    plt.legend()
    plt.show()

    images, masks = load_images(imgs_anns_dict)
    plt.figure(figsize=(20, 10))
    img_index = random.choice(range(len(images)))
    plt.subplot(1,4,1)
    # random_img = img.copy()
    random_img = images[img_index,:,:,0]
    plt.imshow(random_img, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('Lung X-Ray')

    plt.subplot(1,4,2)
    random_mask = masks[img_index,:,:,0]
    plt.imshow(random_mask, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('Mask Ground Truth')

    random_img_2 = np.expand_dims(np.expand_dims(random_img, axis=0), axis=3)
    mask = seg_model.predict(random_img_2)[0][:,:,0] > 0.5
    plt.subplot(1,4,3)
    plt.imshow(mask, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('Predicted Mask')

    plt.subplot(1,4,4)
    plt.imshow(cv2.bitwise_and(random_img, random_img, mask=mask.astype(np.uint8)), cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('Predicted Lung Segmentation')
    plt.savefig('result.png')

    print('---Finish training segmentation model!---')

    # return model
    count = 0

    for root, dir, files in os.walk('./Data/train/'):
        for file in files:
            path = str(root)+'/'+str(file)
            with open(path, 'rb') as f:
                ct_dicom = dcmread(f)
                img = ct_dicom.pixel_array

            img = img.astype(np.float32)
            img = (img / img.max()) * 255

            t = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(128), 
                        transforms.CenterCrop(128),])

            img = np.array(img, dtype=np.float32)
            img = np.array(img)
            img = t(img)
            img = np.array(img)

            img_2 = np.expand_dims(np.expand_dims(img, axis=0), axis=3)
            mask = seg_model.predict(img_2)[0][:,:,0] > 0.5
            img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))

            new_path = './Preprocess_Data/train/' + str(file).split('.dcm')[0] +'.jpg'
            cv2.imwrite(new_path, img)

            print('Complete a picture.')
            count+=1

    print('count: ', count)

if __name__ == '__main__':
    train_segmentation_model()