import os
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Sequential, callbacks
import matplotlib.pyplot as plt

# Get image data
os.makedirs('body_score_dataset', exist_ok=True)
os.system('git clone -b body_scores_prediction_dataset https://github.com/MVet-Platform/M-Vet_Hackathon24.git ./body_score_dataset')

# Load label data
df_train_data = pd.read_csv('body_score_dataset/train_data.csv')

# Get file path for image files
df_train_data['filepath'] = df_train_data.apply(lambda row: glob(f'body_score_dataset/**/{row.filename}')[0], axis=1)

# Create array of body scores and file paths
body_scores = df_train_data.bodyScore.values
file_paths = df_train_data.filepath.values

def load_and_preprocess_image(file_path, body_score=None):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    if body_score is not None:
        return image, body_score
    else:
        return image

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((file_paths, body_scores))
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and split the dataset
dataset = dataset.shuffle(buffer_size=2000)
train_size = int(0.7 * len(file_paths))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Batch the datasets
train_dataset = train_dataset.batch(batch_size=32)
val_dataset = val_dataset.batch(batch_size=32)

# Create model
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    pooling='avg',
)

model = Sequential()
model.add(base_model)

# CONV LAYER ONE
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())

# CONV LAYER TWO
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())

# CONV LAYER THREE
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())

# CONV LAYER FOUR
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())

# GLOBAL POOLING LAYER
model.add(layers.GlobalAveragePooling2D())

# Flatten and fully connected layers
# FLATTENING LAYER
model.add(layers.Flatten())

# DENSE LAYER ONE
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dropout(0.5))

# DENSE LAYER TWO
model.add(layers.Dense(units=1024, activation='relu'))
model.add(layers.Dropout(0.5))

# FINAL LAYER
model.add(layers.Dense(units=1, activation='relu'))

# Compile model
model.compile(loss=losses.mae, optimizer=optimizers.RMSprop())

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

# Train model with callbacks
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)



# Visualize training statistics
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

if 'accuracy' in history.history:
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

plt.show()

# Evaluate the model on the validation dataset
val_loss = model.evaluate(val_dataset)
print(f'Validation Loss: {val_loss}')

# Load submission file
df_submit = pd.read_csv('body_score_dataset/sample_submission.csv')

# Prepare test dataset
df_submit['filepath'] = df_submit.apply(lambda row: glob(f'body_score_dataset/**/{row.filename}')[0], axis=1)
df_submit_file_paths = df_submit.filepath.values
test_dataset = tf.data.Dataset.from_tensor_slices(df_submit_file_paths)
test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(32)

# Make predictions
predictions = model.predict(test_dataset)
predictions_flattened = predictions.flatten()
df_submit['bodyScore'] = [5.0 if i > 5 else i for i in predictions_flattened]

# Save submission file
df_submit[['filename', 'bodyScore']].to_csv('submission.csv', index=False)
