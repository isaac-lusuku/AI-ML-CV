import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging as lg
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# setting the logging messages to error
logger = tf.get_logger()
logger.setLevel(lg.ERROR)

# initiating the training data
celsius_array = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_array = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# building the layers
l1 = tf.keras.layers.Dense(units=1, input_shape=[1])

# building the model
model = tf.keras.Sequential([l1])

# compiling the model
model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))

# training the model
history = model.fit(celsius_array, fahrenheit_array, epochs=500, verbose=False)
if history:
    print("training successful")

# displaying te model convergence with matplotlib
plt.xlabel("number of epochs")
plt.ylabel("loss magnitude")
plt.plot(history.history["loss"])
plt.show()

# print the weights and biases
print(f"the layer values are {l1.get_weights()}")

# predicting from the console
stop = False
while not stop:
    cel_deg = input("enter degrees celsius or any char to stop\n")
    try:
        if type(int(cel_deg)) == int:
            print(f"degrees far is {model.predict([int(cel_deg)])}")
    except ValueError:
        stop = True
