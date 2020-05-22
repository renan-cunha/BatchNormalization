import numpy as np
from typing import Tuple


class LinearLayer:

    def __init__(self, shape: Tuple[int, int]):
        self.weights = np.random.randn(shape[0], shape[1]).astype("float32")
        self.biases = np.random.randn(shape[1], 1).astype("float32")

    def forward(self, x: np.ndarray):
        self.x = x
        self.next_x = (self.x @ self.weights) + self.biases.T
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray):
        self.grad_weights = self.x.T @ next_grad_inputs
        self.grad_biases = np.sum(next_grad_inputs, axis=0, keepdims=True).T
        return next_grad_inputs @ self.weights.T


class SqrErrorLayer:

    def __init__(self, y: np.ndarray):
        self.y = y

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.next_x = np.square(x-self.y)
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        grad_inputs = -(self.y - self.x)*2*next_grad_inputs
        return grad_inputs


class SoftmaxLayer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.next_x = x/(np.sum(x, axis=1, keepdims=True))
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        return next_grad_inputs * (self.next_x * (1 - self.next_x))


class CrossEntropyLoss:

    def __init__(self, y: np.ndarray):
        self.epsilon = 10**-7
        self.y = np.clip(y, self.epsilon, 1.0-self.epsilon)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.p = np.clip(x, self.epsilon, 1.0-self.epsilon)
        return -(self.y*np.log(self.p) + (1-self.y)*np.log(1-self.p))

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        layer_error = (self.p - self.y) / (self.p - self.p**2)
        return layer_error * next_grad_inputs


class SigmoidLayer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.next_x = 1/(np.exp(-x)+1)
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        return (self.next_x*(1-self.next_x))*next_grad_inputs


