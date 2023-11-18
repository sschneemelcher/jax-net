from net import NeuralNet, get_num_of_neurons_in_architecture
import keras
import numpy as np


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)


architecture = [x_train.shape[1], 64, 10]
dna = np.random.normal(0, 0.1, size=(get_num_of_neurons_in_architecture(architecture)))

net = NeuralNet(dna, architecture)

epochs = 100
for e in range(epochs):
    print(f"epoch {e}: acc: {net.fit(x_train[:5000], y_train[5000], 100, 0.1, 0.05)}")
