import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Load the previously saved model
model = tf.keras.models.load_model('Waste-classifier.h5')

# Define the paths to the new training and validation datasets
new_training_path = r"C:\Users\Administrator\Desktop\ML\Waste segmentation\waste_resources\FINAL_TEST"

# Define the image data generators with the same preprocessing as before
training_data_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
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

validation_data_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    validation_split=0.2,
    rescale=1./255
)

# creating the generator for the training dataset
new_training_data = training_data_generator.flow_from_directory(
    new_training_path,
    target_size=(224, 224),
    batch_size=128,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=43
)

# creating the generator for the validation dataset (no augmentation)
new_validation_data = validation_data_generator.flow_from_directory(
    new_training_path,
    target_size=(224, 224),
    shuffle=True,
    seed=43,
    subset='validation',
    batch_size=128,
    class_mode='categorical'
)


# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training the model on the new data
history = model.fit(
    new_training_data,
    steps_per_epoch=len(new_training_data),  # Adjust based on the size of your data
    epochs=10,  # Set the number of epochs for retraining
    validation_data=new_validation_data,
    validation_steps=len(new_validation_data),
    callbacks=[early_stopping, reduce_lr]
)

# Save the retrained model if needed
model.save('Waste-classifier-retrained-specific-class.h5')

# Plotting training and validation accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()
