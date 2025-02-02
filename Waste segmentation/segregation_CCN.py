import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input

# the os paths to the training and the testing datasets
training_path = r"C:\Users\Administrator\Desktop\ML\Waste segmentation\waste_resources\DATASET\TRAIN"
testing_path = r"C:\Users\Administrator\Desktop\ML\Waste segmentation\waste_resources\DATASET\TEST"

# defining the image data generator for training data with augmentation
training_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# defining the image data generator for the validation data
validation_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rescale=1./255
)

# defining the image data generator for the testing data
test_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255
)

# creating the generator for the training dataset
training_data = training_data_generator.flow_from_directory(
    training_path,
    target_size=(224, 224),
    batch_size=128,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=43
)

# creating the generator for the validation dataset (no augmentation)
validation_data = validation_data_generator.flow_from_directory(
    training_path,
    target_size=(224, 224),
    shuffle=True,
    seed=43,
    subset='validation',
    batch_size=128,
    class_mode='categorical'
)

# creating the generator for the test dataset ( no augmentation)
test_data = test_data_generator.flow_from_directory(
    testing_path,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=128,
    shuffle=True,
)

# building the model
model = Sequential()

# Convolutional Layer 1
model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.3))

# Convolutional Layer 2
model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.3))

# Convolutional Layer 3
model.add(Conv2D(256, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.3))

# Convolutional Layer 4
model.add(Conv2D(512, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.3))

# Convolutional Layer 5
model.add(Conv2D(512, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.3))

# Flatten Layer
model.add(Flatten())

# Fully Connected Layer 1
model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.4))

# Fully Connected Layer 2
model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.4))

# Fully Connected Layer 3
model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.4))

# Output Layer
model.add(Dense(2))
model.add(Activation("softmax"))

# Compile the model
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])

# Train the model
history = model.fit(
    training_data,
    steps_per_epoch=training_data.samples // training_data.batch_size,
    epochs=15,
    validation_data=validation_data,
    validation_steps=validation_data.samples // validation_data.batch_size
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, steps=test_data.samples // test_data.batch_size)
print('Test accuracy:', test_acc)

model.save('Waste-classifier.h5', overwrite=True, include_optimizer=True, save_format='h5')