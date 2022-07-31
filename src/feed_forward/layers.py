import numpy as np
from abc import ABC, abstractmethod


epsilon = 10 ** -3


class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        pass


class ParamLayer(Layer, ABC):

    @abstractmethod
    def apply_gradients(self, learning_rate: float) -> None:
        pass


class LinearLayer(ParamLayer):

    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.random.normal(0.0, 0.1,
                                        size=input_dim * output_dim)
        self.weights = self.weights.reshape(input_dim, output_dim)
        self.weights = self.weights.astype("float32")
        self.biases = np.zeros((1, output_dim)).astype("float32")

        # forward params
        self.x = np.zeros(0)

        # backward params
        self.grad_biases = np.zeros(0)
        self.grad_weights = np.zeros(0)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.x = x
        return (self.x @ self.weights) + self.biases

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        self.grad_weights = self.x.T @ grad_input
        self.grad_biases = np.sum(grad_input, axis=0, keepdims=True)
        return grad_input @ self.weights.T

    def apply_gradients(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases


class SqrErrorLayer(Layer):

    def __init__(self, y: np.ndarray):
        self.y = y

        # forward params
        self.x = np.zeros(0)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.x = x
        return np.square(x - self.y)

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        return -2 * (self.y - self.x) * grad_input


class SoftmaxLayer(Layer):

    def __init__(self) -> None:
        self.next_x = np.zeros(0)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.next_x = np.exp(x) / np.sum(np.exp(x) + 0.0001, axis=1, keepdims=True)
        return self.next_x

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        softmax = self.next_x.reshape(-1,1)
        return grad_input * (np.diagflat(softmax) - np.dot(softmax, softmax.T))


class CrossEntropyLoss(Layer):

    def __init__(self, y: np.ndarray):
        self.y = np.clip(y, epsilon, 1.0 - epsilon)

        # forward params
        self.p = np.zeros(0)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.p = np.clip(x, epsilon, 1.0 - epsilon)
        return -(self.y * np.log(self.p) + (1-self.y) * np.log(1 - self.p))

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        layer_error = (self.p - self.y) / (self.p - self.p ** 2)
        return layer_error * grad_input


class BatchNormLayer(ParamLayer):

    def __init__(self, dims: int) -> None:
        self.gamma = np.ones((1, dims), dtype="float32")
        self.bias = np.zeros((1, dims), dtype="float32")

        self.running_mean_x = np.zeros(0)
        self.running_var_x = np.zeros(0)

        # forward params
        self.var_x = np.zeros(0)
        self.stddev_x = np.zeros(0)
        self.x_minus_mean = np.zeros(0)
        self.standard_x = np.zeros(0)
        self.num_examples = 0
        self.mean_x = np.zeros(0)
        self.running_avg_gamma = 0.9

        # backward params
        self.gamma_grad = np.zeros(0)
        self.bias_grad = np.zeros(0)

    def update_running_variables(self) -> None:
        is_mean_empty = np.array_equal(np.zeros(0), self.running_mean_x)
        is_var_empty = np.array_equal(np.zeros(0), self.running_var_x)
        if is_mean_empty != is_var_empty:
            raise ValueError("Mean and Var running averages should be "
                             "initilizaded at the same time")
        if is_mean_empty:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x
        else:
            gamma = self.running_avg_gamma
            self.running_mean_x = gamma * self.running_mean_x + \
                                  (1.0 - gamma) * self.mean_x
            self.running_var_x = gamma * self.running_var_x + \
                                 (1. - gamma) * self.var_x

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.num_examples = x.shape[0]
        if train:
            self.mean_x = np.mean(x, axis=0, keepdims=True)
            self.var_x = np.mean((x - self.mean_x) ** 2, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            self.mean_x = self.running_mean_x.copy()
            self.var_x = self.running_var_x.copy()

        self.var_x += epsilon
        self.stddev_x = np.sqrt(self.var_x)
        self.x_minus_mean = x - self.mean_x
        self.standard_x = self.x_minus_mean / self.stddev_x
        return self.gamma * self.standard_x + self.bias

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        standard_grad = grad_input * self.gamma

        var_grad = np.sum(standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3/2),
                          axis=0, keepdims=True)
        stddev_inv = 1 / self.stddev_x
        aux_x_minus_mean = 2 * self.x_minus_mean / self.num_examples

        mean_grad = (np.sum(standard_grad * -stddev_inv, axis=0,
                            keepdims=True) +
                            var_grad * np.sum(-aux_x_minus_mean, axis=0,
                            keepdims=True))

        self.gamma_grad = np.sum(grad_input * self.standard_x, axis=0,
                                 keepdims=True)
        self.bias_grad = np.sum(grad_input, axis=0, keepdims=True)

        return standard_grad * stddev_inv + var_grad * aux_x_minus_mean + \
               mean_grad / self.num_examples

    def apply_gradients(self, learning_rate: float) -> None:
        self.gamma -= learning_rate * self.gamma_grad
        self.bias -= learning_rate * self.bias_grad


class SigmoidLayer(Layer):

    def __init__(self) -> None:
        # forward params
        self.next_x = np.zeros(0)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.next_x = 1 / (np.exp(-x) + 1)
        return self.next_x

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        return (self.next_x * (1 - self.next_x)) * grad_input

