import os
import glob
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

"""
    SECTION ONE --> LOADING THE IMAGE DATA AND PREPARING IT INTO A DATAFRAME
"""
dataset_path = "C:/Users/Administrator/Desktop/ML/drowsy detection"  # path to our dataset

# Dictionary to store image file paths for each class
classes = ['Drowsy', 'Non Drowsy']
data_dict = {class_name: glob.glob(os.path.join(dataset_path, class_name, "*.*")) for class_name in classes}

# Print out how many images in each class
for class_name, files in data_dict.items():
    print(f"{class_name}: {len(files)} images")

# Dictionaries to  DataFrame
data_list = []
for class_name, files in data_dict.items():
    for f in files:
        data_list.append({'filepath': f, 'label': class_name})

# creating the dataframe and demonstration
df = pd.DataFrame(data_list)
print(df.tail())
print("Total images:", len(df))


"""
    SECTION TWO --> BAR GRAPH TO DEMONSTRATE THE DISTRIBUTION OF THE DATA USING SEABORN
"""
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df, palette='Set2')
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
plt.title("Number of Images per Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


"""
    SECTION THREE --> DISPLAYING SAMPLE IMAGES FROM OUR DATASET
"""


def display_random_images(df, label, n=6):
    # Display n random images from the specified class label
    sample_paths = df[df['label'] == label]['filepath'].sample(n).values
    plt.figure(figsize=(15, 8))
    for i, img_path in enumerate(sample_paths):
        img = cv2.imread(img_path)
        # OpenCV reads in BGR format, convert it to RGB for proper display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, i+1)
        plt.imshow(img_rgb)
        plt.title(f"{label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Display random images from each class
for class_name in classes:
    print(f"Random samples for class: {class_name}")
    display_random_images(df, class_name, n=6)

"""
    SECTION FOUR --> PIE CHART OF THE DATA
"""
# First, compute the count of images per class from the DataFrame
class_counts = df['label'].value_counts()
print(class_counts)

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'], shadow=True)
plt.title('Distribution of Classes in the Driver Drowsiness Dataset')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

"""
    SECTION FIVE --> COMPUTING AND DISPLAYING THE MEAN AND STANDARD DEVIATIONS
"""


def compute_image_stats(image_paths, num_samples=100):
    # Compute the average pixel value and std deviation for a sample of images.
    means = []
    stds = []
    for img_path in random.sample(image_paths, min(num_samples, len(image_paths))):
        img = cv2.imread(img_path)
        # Normalize image: convert pixel values to float in [0,1]
        img = img.astype('float32') / 255.0
        means.append(np.mean(img))
        stds.append(np.std(img))
    return np.mean(means), np.mean(stds)


# creating a stats dictionary to help us visualize the mean
stats = {}
for class_name, paths in data_dict.items():
    mean_val, std_val = compute_image_stats(paths)
    stats[class_name] = {'mean': mean_val, 'std': std_val}
    print(f"{class_name}: Mean = {mean_val:.3f}, Std = {std_val:.3f}")

# Plotting the image  stats
stats_df = pd.DataFrame(stats).T.reset_index().rename(columns={'index': 'class'})
plt.figure(figsize=(8, 6))
sns.barplot(x='class', y='mean', data=stats_df, palette='pastel')
plt.title("Average Image Intensity per Class")
plt.ylabel("Average Pixel Value")
plt.show()