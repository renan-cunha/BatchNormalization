import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_inputs: np.ndarray) -> np.ndarray:
        pass


class ParamLayer(Layer, ABC):

    @abstractmethod
    def apply_gradients(self, learning_rate: float) -> None:
        pass


class LinearLayer(ParamLayer):

    def __init__(self, shape: Tuple[int, int]):
        self.weights = np.random.randn(shape[0], shape[1]).astype("float32")
        self.biases = np.random.randn(shape[1], 1).astype("float32")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.next_x = (self.x @ self.weights) + self.biases.T
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        self.grad_weights = self.x.T @ next_grad_inputs
        self.grad_biases = np.sum(next_grad_inputs, axis=0, keepdims=True).T
        return next_grad_inputs @ self.weights.T

    def apply_gradients(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases


class SqrErrorLayer(Layer):

    def __init__(self, y: np.ndarray):
        self.y = y

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.next_x = np.square(x-self.y)
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        grad_inputs = -(self.y - self.x)*2*next_grad_inputs
        return grad_inputs


class SoftmaxLayer(Layer):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.next_x = x/(np.sum(x, axis=1, keepdims=True))
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        return next_grad_inputs * (self.next_x * (1 - self.next_x))


class CrossEntropyLoss(Layer):

    def __init__(self, y: np.ndarray):
        self.epsilon = 10**-7
        self.y = np.clip(y, self.epsilon, 1.0-self.epsilon)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.p = np.clip(x, self.epsilon, 1.0-self.epsilon)
        return -(self.y*np.log(self.p) + (1-self.y)*np.log(1-self.p))

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        layer_error = (self.p - self.y) / (self.p - self.p**2)
        return layer_error * next_grad_inputs


class BatchNormLayer(ParamLayer):

    def __init__(self, dims: int) -> None:
        self.gamma = np.random.uniform(low=10**-7, size=dims).reshape(-1, 1)
        self.bias = np.zeros(dims).reshape(-1, 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.mean_x = np.mean(x, axis=0, keepdims=True)
        self.x_minus_mean = self.x - self.mean_x
        self.var_x = np.mean(self.x_minus_mean**2, axis=0, keepdims=True)
        self.var_x += 10**-7
        self.stddev_x = np.sqrt(self.var_x)

        self.standard_x = (self.x_minus_mean)/self.stddev_x
        self.rescaled_x = self.gamma.T*self.standard_x + self.bias.T
        return self.rescaled_x

    def backward(self, input_grads: np.ndarray) -> np.ndarray:
        standard_grad = input_grads * self.gamma.T

        var_grad = np.sum(standard_grad * self.x_minus_mean * (-1/2)*(self.var_x)**(-3/2), axis=0, keepdims=True)
        stddev_inv = 1/(self.stddev_x)
        aux_x_minus_mean = 2*(self.x_minus_mean)/self.x.shape[0]

        mean_grad = (standard_grad @ (-1*stddev_inv.T) +
                    var_grad * np.mean(aux_x_minus_mean, axis=0, keepdims=True))
        

        self.gamma_grad = np.sum(input_grads * self.standard_x, axis=0, 
                                 keepdims=True).T
        self.bias_grad = np.sum(input_grads, axis=0, keepdims=True).T
        return input_grads * stddev_inv + var_grad*aux_x_minus_mean +\
               mean_grad/self.x.shape[0]

    def apply_gradients(self, learning_rate: float) -> None:
        self.gamma -= learning_rate*self.gamma_grad
        self.bias -= learning_rate*self.bias_grad


class SigmoidLayer(Layer):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.next_x = 1/(np.exp(-x)+1)
        return self.next_x

    def backward(self, next_grad_inputs: np.ndarray) -> np.ndarray:
        return (self.next_x*(1-self.next_x))*next_grad_inputs

