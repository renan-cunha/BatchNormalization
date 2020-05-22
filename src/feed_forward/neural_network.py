import numpy as np
from typing import List
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
from typing import Tuple, List
from layers import LinearLayer, SigmoidLayer, CrossEntropyLoss
import pandas as pd


class NeuralNetwork:

    def __init__(self, num_neurons: List[int]):
        self.layers = []
        for index, neuron in enumerate(num_neurons[:-1]):
            layer = LinearLayer((neuron, num_neurons[index+1]))
            self.layers.append(layer)
            layer = SigmoidLayer()
            self.layers.append(layer)
        self.loss_layer = None
    
    def backward_propagation(self, output: np.ndarray) -> None:
        grad_input = self.loss_layer.backward(np.ones_like(output)) 
        for layer_index in range(len(self.layers)-1, -1, -1):
            grad_input = self.layers[layer_index].backward(grad_input)

    def update_params(self, learning_rate: float) -> None:
        for layer_index in range(len(self.layers)):
            layer = self.layers[layer_index]
            if type(layer) == LinearLayer:
                self.layers[layer_index].weights -= learning_rate*layer.grad_weights
                self.layers[layer_index].biases -= learning_rate*layer.grad_biases

    def fit(self, x: np.ndarray, y: np.ndarray,
            learning_rate: float, training_iters: int,
            batch_size: int) -> np.array:
        num_examples = x.shape[0]
        losses = np.empty(training_iters)
        random_index = np.linspace(0, num_examples-1, num_examples).astype(int)
        for i in range(training_iters):
            np.random.shuffle(random_index)
            x = x[random_index]
            y = y[random_index]
            for index_batch in range(0, num_examples, batch_size):
                mini_batch_x = x[index_batch: index_batch + batch_size]
                mini_batch_y = y[index_batch: index_batch + batch_size]
                self.loss_layer = CrossEntropyLoss(mini_batch_y)
                y_pred = self.predict(mini_batch_x)
                self.loss_layer.forward(y_pred)
                self.backward_propagation(y_pred)
                self.update_params(learning_rate)
                   
            self.loss_layer = CrossEntropyLoss(y)
            y_pred = self.predict(x)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_labels = np.argmax(y, axis=1)

            accuracy = np.sum(y_labels == y_pred_labels) / num_examples

            error = self.loss_layer.forward(y_pred)
            mean_error = np.mean(error)

            print(f"EPOCH {i}")
            print(f"Error: {mean_error} | Accuracy: {accuracy}")

        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == "__main__":
    csv = pd.read_csv("../mnist-in-csv/mnist_train.csv")
    end_index = 60000
    X = csv.values[:end_index, 1:].astype("float64")
    X /= 255
    y = csv.values[:end_index, 0]
    y_one_hot = np.zeros((y.size, y.max()+1))
    y_one_hot[np.arange(y.size), y] = 1
    y_one_hot = y_one_hot.astype("float64")
    
    model = NeuralNetwork([784, 100, 100, 100, 10])
    model.fit(X, y_one_hot, training_iters=50, learning_rate=0.001, batch_size=60)
    csv = pd.read_csv("../mnist-in-csv/mnist_test.csv")
    X = csv.values[:, 1:].astype("float64")
    X /= 255
    y = csv.values[:, 0]
    y_one_hot = np.zeros((y.size, y.max()+1))
    y_one_hot[np.arange(y.size), y] = 1
    y_one_hot = y_one_hot.astype("float64")

    print(np.sum(np.argmax(model.predict(X), axis=1) == y)/10000)

