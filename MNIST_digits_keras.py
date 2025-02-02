import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# getting ths dataset
dset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = dset.load_data()

# normalizing the data
x_train, x_test = x_train/255, x_test/255

# creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(units=10)
])

# compiling the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training the model
batch_size = 64
epochs = 20

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

# displaying the model conversion
if history:
    plt.xlabel("number of epochs")
    plt.ylabel("loss magnitude")
    plt.plot(history.history["loss"])
    plt.show()

# evaluating the model
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)


# testing function with a random sample
def test_func(n):
    prob_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    prediction = prob_model.predict(x_test)[n]
    print(f"real value is {y_test[n]}")
    print(f"predicted value is {np.argmax(prediction)}")


# you can call this with a random index
"""
test_func(n)
"""

