from jax._src.typing import Array
import jax.numpy as jnp
from jax import  jit, random
from random import randint


def __relu(x: Array) -> Array:
    return jnp.maximum(x, 0)


relu = jit(__relu)


def __softmax(x: Array) -> Array:
    exp = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    s = jnp.sum(exp, axis=-1, keepdims=True)
    return exp / s


softmax = jit(__softmax)


class DenseLayer:
    def __init__(
        self,
        weights: Array,
        biases: Array,
    ):
        self.weights = weights
        self.biases = biases
        self.units, self.input_size = weights.shape




class NeuralNet:
    def __init__(self, dna: Array, architecture: list[int]) -> None:
        self.dna = dna
        self.architecture = architecture
        self.layers = self.__build_layers_from_dna()
        self.predict = jit(self.__predict)

    def __predict(self, x):
        for i in range(len(self.layers) - 1):
            x = relu(jnp.dot(self.layers[i].weights, x) + self.layers[i].biases)

        return softmax(jnp.dot(self.layers[-1].weights, x) + self.layers[-1].biases)


    def __build_layers_from_dna(self) -> list[DenseLayer]:

        layers = []
        counter = 0

        for i in range(len(self.architecture) - 1):
            c1 = counter + self.architecture[i] * self.architecture[i+1]
            c2 = counter + self.architecture[i] * self.architecture[i+1] + self.architecture[i+1]
            layer = DenseLayer(dna[counter:c1].reshape((self.architecture[i+1], self.architecture[i])), dna[c1:c2].reshape((self.architecture[i+1], 1)))
            layers.append(layer)
            counter = c2

        return layers

    def fit(self, x: Array, y: Array):
        n = 100
        noise = random.normal(random.PRNGKey(randint(0, 999999999)), [n, len(dna)])

        for i in range(n):
            print(noise[i])
            net = NeuralNet(self.dna + noise[i], self.architecture)
            preds = net.predict(x)
            print(preds)
            #print(jnp.mean(preds == y))


        

def get_num_of_neurons_in_architecture(architecture: list[int]) -> int:
    n = 0
    for i in range(len(architecture) - 1):
        n += architecture[i] * architecture[i+1] + architecture[i]

    return n

architecture = [10, 64, 32, 10]
key = random.PRNGKey(randint(0, 999999999))
dna = random.normal(key, [get_num_of_neurons_in_architecture(architecture)])

# net = NeuralNet(dna, architecture)
