import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

from data_preprocessing import *

def Resnet50_model(input_shape, train_generator, validation_generator,class_weights ,num_classes, model_path):
    # Finetuning the full-network of the pretrained model with the newly added layers first
    # Load pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = True
    # Add layers at the end
    X = base_model.output
    X = Flatten()(X)

    X = Dense(512, kernel_initializer='he_uniform')(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(32, kernel_initializer='he_uniform')(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    output = Dense(num_classes, activation='softmax')(X)

    model = Model(inputs=base_model.input, outputs=output)
    # initializing the optimizer of the model (Adam) to compile the model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])
    # Callbacks (Early Stop and Learning rate reduction)
    early_stop_1 = EarlyStopping(monitor='val_loss', patience=5, verbose=1, 
                        mode='auto', restore_best_weights=True)
    early_stop_2 = EarlyStopping(monitor='val_loss', patience=15, verbose=1, 
                        mode='auto', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                        verbose=1, mode='auto')
    # first training
    Resnet_history1 = model.fit(train_generator,
                              validation_data=validation_generator,
                              epochs=20,
                              shuffle=True,
                              verbose=1,
                              callbacks=[reduce_lr, early_stop_1],
                              class_weight=class_weights
                             )
    
    # freezing the core resnetlayers and training only the first 50 layers of the Resnet models
    # (benefiting from the pre-trained weights and learned features of the base model). 
    # This approach helps balance the adaptation to the new task (Artist Art-Pieces images) 
    # with the preservation of valuable pre-trained features.
    # Freeze core ResNet layers and train again 
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[:50]:
        layer.trainable = True

    optimizer = Adam(learning_rate=0.0001)

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer, 
                metrics=['accuracy'])

    Resnet_history2 = model.fit(train_generator,
                                validation_data=validation_generator,
                                epochs=50,
                                shuffle=True,
                                verbose=1,
                                callbacks=[reduce_lr, early_stop_2],
                                class_weight=class_weights
                                )
    model.save(model_path)
    return model


# Usage Guide 
artists_csv = 'Artist_data/artists.csv'
images_dir_path = 'Artist_data/images/images'
input_shape, train_generator, validation_generator , class_weights, num_classes = full_data_preprocessing_pipeline(32, 224, 224, 3, artists_csv, images_dir_path)
model_path = 'model/Artist_Resnet_model.h5'
Resnet50_model(input_shape, train_generator, validation_generator,class_weights ,num_classes, model_path)