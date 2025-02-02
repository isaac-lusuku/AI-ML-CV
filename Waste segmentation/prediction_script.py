import tensorflow as tf
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

waste_model = tf.keras.models.load_model('Waste-classifier-retrained-specific-class.h5')

# the os paths to the training datasets
testing_path_O = r"C:\Users\Administrator\Desktop\ML\Waste segmentation\waste_resources\REAL_SAMPLES"
testing_path_R = r"C:\Users\Administrator\Desktop\ML\Waste segmentation\waste_resources\LATEST_TEST\R"

# loading the images form the folders
images_O = [f for f in os.listdir(testing_path_O) if os.path.isfile(os.path.join(testing_path_O, f))]
images_R = [f for f in os.listdir(testing_path_R) if os.path.isfile(os.path.join(testing_path_R, f))]

class_labels = ["biodegradable", "non-biodegradable"]


# a function for predicting and illustrating the data
def predict_and_plot(img_list, folder_path):

    plt.figure(figsize=(10, 10))

    for img_file in img_list:
        img_path = os.path.join(folder_path, img_file)

        # preprocessing the images
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        clipped = img_array / 255.

        img_array = preprocess_input(img_array)

        # predicting the class
        prediction = waste_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_labels[predicted_class]

        print(f"Image: {img_file}, Prediction: {prediction}, argmax: {np.argmax(prediction, axis=1)[0]}")

        # plotting the test images
        plt.subplot(2, 5, img_list.index(img_file)+1)
        plt.imshow(clipped[0])
        plt.title(predicted_class_name)
        plt.axis('off')
    plt.show()


# Randomly select ten images
selected_images = random.sample(images_R, 10)
# selected_images = images_R[10:19]
predict_and_plot(selected_images, testing_path_R)


# test_data_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1./255)
# test_data = test_data_generator.flow_from_directory(
#     r"C:\Users\Administrator\Desktop\ML\Waste segmentation\waste_resources\DATASET\TEST",
#     target_size=(224, 224),
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=False
# )
#
# test_loss, test_acc = waste_model.evaluate(test_data)
# print('Test accuracy:', test_acc)
