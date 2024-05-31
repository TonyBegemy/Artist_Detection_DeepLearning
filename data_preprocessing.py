import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def read_csv(csv_path):
    return pd.read_csv(csv_path)

def sort_artist_paintings(df, paintings_number):
    artists = df.sort_values(by=['paintings'], ascending=False)
    artists_top = df[df['paintings'] >= paintings_number].reset_index(drop=True)
    return artists_top

def class_weights_dict(artists_top):
    artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
    artists_top = artists_top.sort_values(by='paintings', ascending=False).reset_index(drop=True)
    artists_top = artists_top[['name', 'paintings', 'class_weight']]
    class_weights = artists_top['class_weight'].to_dict()
    return class_weights

def get_artist_names(artists_top):
    artists_top_name = artists_top['name'].str.replace(' ', '_').values
    artists_top_name[7] = "Albrecht_Du╠êrer"
    return artists_top_name


def data_augmentation():
    train_datagen = ImageDataGenerator(
    rescale=1.0/255, # rescaling the pixel values to [0,1]
    rotation_range=20, # rotating images in the range ( 0 to 20 degrees)
    width_shift_range=0.2, # horizontal shift the image to the left or right
    height_shift_range=0.2, # vertical shift the image to the left or right
    shear_range=0.2, # shear transformation in the range [0, 0.2]
    zoom_range=0.2, # zooming inside pictures by up to 20%
    horizontal_flip=True, # flipping images horizontally
    fill_mode='nearest', # a strategy for filling in pixels apear after a rotation or shift 
    validation_split=0.2  # splitting data into 80% training and 20% validation (to test)
    )
    return train_datagen

def train_validation_generators(batch_size, img_height, img_width, channels, image_folder, artists_top_name, train_datagen):
    input_shape = (img_height, img_width, channels)
    num_classes = len(artists_top_name)
    # creating train and validation generators to preprocess images directly to the models
    train_generator = train_datagen.flow_from_directory(
        image_folder,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=artists_top_name.tolist()
    )

    validation_generator = train_datagen.flow_from_directory(
        image_folder,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        classes=artists_top_name.tolist()
    )
    return train_generator, validation_generator, num_classes, input_shape


def full_data_preprocessing_pipeline(batch_size, img_height, img_width, channels, artists_csv, images_dir_path):
    # read csv file artists.csv
    df = read_csv(artists_csv)
    artists_top = sort_artist_paintings(df, 150)
    artists_top_name = get_artist_names(artists_top)
    class_weights = class_weights_dict(artists_top)
    train_datagen = data_augmentation()
    train_generator, validation_generator, num_classes, input_shape = train_validation_generators(batch_size, img_height, img_width, channels, images_dir_path, artists_top_name, train_datagen)
    return input_shape, train_generator, validation_generator, class_weights, num_classes

# Test Code
artists_csv = 'Artist_data/artists.csv'
images_dir_path = 'Artist_data/images/images'
input_shape, train_generator, validation_generator , class_weights, num_classes  = full_data_preprocessing_pipeline(64, 224, 224, 3, artists_csv, images_dir_path)





