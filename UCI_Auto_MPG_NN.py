import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

# loading the dataset
auto_MPG = fetch_ucirepo(id=9)
dataset = auto_MPG.data.original

# cleaning the dataset
dataset.pop("car_name")

popped = dataset.pop("origin")
dataset["USA"] = (popped == 1)*1
dataset["JAPAN"] = (popped == 2)*1
dataset["EUROPE"] = (popped == 3)*1

dataset.dropna(inplace=True)

# splitting the dataset into training and testing
training_dataset = dataset.sample(frac=0.8, random_state=0)
testing_dataset = dataset.drop(training_dataset.index)

# splitting the training and testing into features and labels(MPG)
y_training_dataset = training_dataset.pop("mpg")
x_training_dataset = training_dataset
y_testing_dataset = testing_dataset.pop("mpg")
x_testing_dataset = testing_dataset

# creating the normalization layer for the training dataset
normalizer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[9])
normalizer.adapt(np.array(x_training_dataset))
print(x_training_dataset.describe().transpose())
print(normalizer.mean.numpy())

# initializing the model
multi_model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

# compiling the model
loss = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = ["mean_absolute_error"]
multi_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# training the model
multi_history = multi_model.fit(
    x_training_dataset,
    y_training_dataset,
    verbose=2,
    epochs=100
)
if multi_history:
    print("the training was successful")


"""
-------->    FROM THIS POINT AM BUILDING THE LINEAR REGRESSION MODEL FOR THE MPG AND AN SINGLE FEATURE   <--------
"""


def linear_regression_model(feature):

    """___start by visualizing the relationship___"""
    plt.scatter(x_training_dataset[feature], y_training_dataset)
    plt.xlabel(feature)
    plt.ylabel("MPG")
    plt.show()
    print("visualizing successful")

    """___normalizing the feature___"""
    feature_normalizer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[1])
    feature_normalizer.adapt(x_training_dataset[feature])
    print("normalizing successful")

    """___initializing the model___"""
    # we are training a deep NN
    model = tf.keras.Sequential([
        feature_normalizer,
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=1)
    ])
    print("initialization successful")

    """___compiling the model___"""
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print("compiling successful")

    """___training the model___"""
    history = model.fit(
        x_training_dataset[feature],
        y_training_dataset.to_numpy(dtype=float),
        verbose=2,
        epochs=100,
        # batch_size=10,
        validation_split=0.2
    )
    if history:
        print("training successful")


"""
uncomment to call the function
"""
# linear_regression_model("horsepower")






