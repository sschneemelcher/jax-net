from typing import Any
from jax._src.typing import Array
import jax.numpy as jnp
from jax import jit, random
from random import randint
import numpy as np


@jit
def relu(x: Array) -> Array:
    return jnp.maximum(x, 0)


@jit
def softmax(x: Array) -> Array:
    exp = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    s = jnp.sum(exp, axis=-1, keepdims=True)
    return exp / s


@jit
def get_num_of_neurons_in_architecture(architecture: list[int]) -> int:
    n = 0
    for i in range(len(architecture) - 1):
        n += architecture[i] * architecture[i + 1] + architecture[i]

    return n


class DenseLayer:
    def __init__(
        self,
        weights: np.ndarray[Any, np.dtype[np.float64]],
        biases: np.ndarray[Any, np.dtype[np.float64]],
    ):
        self.weights = weights
        self.biases = biases
        self.units, self.input_size = weights.shape


class NeuralNet:
    def __init__(self, dna: np.ndarray[Any, np.dtype[np.float64]], architecture: list[int]) -> None:
        self.dna = dna
        self.architecture = architecture
        self.layers = self.__build_layers_from_dna()

    def __build_layers_from_dna(self) -> list[DenseLayer]:
        layers = []
        counter = 0

        for i in range(len(self.architecture) - 1):
            c1 = counter + self.architecture[i] * self.architecture[i + 1]
            c2 = counter + self.architecture[i] * self.architecture[i + 1] + self.architecture[i + 1]
            layer = DenseLayer(
                self.dna[counter:c1].reshape((self.architecture[i], self.architecture[i + 1])),
                self.dna[c1:c2].reshape((1, self.architecture[i + 1])),
            )
            layers.append(layer)
            counter = c2

        return layers

    def predict(self, x):
        for i in range(len(self.layers) - 1):
            x = relu(jnp.dot(x, self.layers[i].weights) + self.layers[i].biases)

        return softmax(jnp.dot(x, self.layers[-1].weights) + self.layers[-1].biases)

    def fit(self, x: Array, y: Array, n: int = 300, sigma: float = 0.1, lr: float = 0.01):
        noise = np.random.normal(0, sigma, size=(n, len(self.dna)))

        R = np.zeros(n)

        for i in range(n):
            net = NeuralNet(self.dna + noise[i], self.architecture)
            preds = jnp.argmax(net.predict(x), axis=1)
            R[i] = jnp.mean(preds == y)

        std = np.std(R)

        # if std == 0 we cannot improve from the scores we collected
        if std != 0:
            A = (R - np.mean(R)) / std
            self.dna = self.dna + lr / (n * sigma) * np.dot(noise.T, A)

        return R.mean()
