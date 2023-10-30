from jax._src.typing import Array
import jax.numpy as jnp
from jax import jit, random
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
    def __init__(self, units: int, input_size: int):
        self.key = random.PRNGKey(randint(0, 999999999))
        self.units = units
        self.input_size = input_size
        self.weights = random.normal(self.key, (units, input_size))
        self.biases = random.normal(self.key, (units, 1))



class NeuralNet():
    def __init__(self, layers: list[DenseLayer]) -> None:
        self.layers = layers
        self.predict = jit(self.__predict)

    def __predict(self, x):
        for i in range(len(self.layers) - 1):
            x = relu(jnp.dot(self.layers[i].weights, x) + self.layers[i].biases)

        return softmax(jnp.dot(self.layers[-1].weights, x) + self.layers[-1].biases)

# architecture = [
#     DenseLayer(units=64, input_size=10),
#     DenseLayer(units=32, input_size=64),
#     DenseLayer(units=10, input_size=32),
# ]
