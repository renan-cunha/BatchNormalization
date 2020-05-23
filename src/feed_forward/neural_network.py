import numpy as np
from typing import List
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
from typing import Tuple, List
from src.feed_forward.layers import LinearLayer, SigmoidLayer, CrossEntropyLoss, ParamLayer, Layer, BatchNormLayer
import pandas as pd
from tqdm import tqdm
import math


class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers[:]
        self.loss_layer = None
    
    def backward_propagation(self, output: np.ndarray) -> None:
        grad_input = self.loss_layer.backward(np.ones_like(output)) 
        for layer_index in range(len(self.layers)-1, -1, -1):
            grad_input = self.layers[layer_index].backward(grad_input)

    def update_params(self, learning_rate: float) -> None:
        for layer_index in range(len(self.layers)):
            layer = self.layers[layer_index]
            if isinstance(layer, ParamLayer):
                layer.apply_gradients(learning_rate)

    def evaluate_model(self, test_x: np.ndarray, 
                       test_y: np.ndarray) -> Tuple[float, float]:
        """Returns loss and accuracy in the test set"""
        self.loss_layer = CrossEntropyLoss(test_y)
        y_pred = self.predict(test_x)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_labels = np.argmax(test_y, axis=1)
        loss = self.loss_layer.forward(y_pred)
        accuracy = np.sum(y_labels == y_pred_labels) / test_x.shape[0]
        return np.mean(loss), accuracy

    def fit(self, x: np.ndarray, y: np.ndarray,
            learning_rate: float, steps: int,
            batch_size: int, test_x: np.ndarray, 
            test_y: np.ndarray) -> np.array:
        
        num_examples = x.shape[0]
        num_batches = math.ceil(num_examples / batch_size)
        metrics = np.zeros((11, 2))
        random_index = np.linspace(0, num_examples-1, num_examples).astype(int)
        step = 0
        pbar = tqdm(total=steps)
        metrics_index = 0
        while step < steps:
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

                step += 1
                pbar.update(1)
                if step > steps:
                    break
            
            loss, accuracy = self.evaluate_model(x, y)
            metrics[metrics_index, :] = loss, accuracy
            metrics_index += 1
            print(f"Step {step}")
            print(f"Loss: {loss} | Accuracy: {accuracy}")                   

        return metrics

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == "__main__":
    
    
    def pre_process_data(location: str) -> Tuple[np.ndarray, np.ndarray]:
        csv_train = pd.read_csv(location)
        X = csv_train.values[:, 1:].astype("float32")
        X /= 255
        y = csv_train.values[:, 0]
        y_one_hot = np.zeros((y.size, y.max()+1))
        y_one_hot[np.arange(y.size), y] = 1
        y_one_hot = y_one_hot.astype("float32")
        return X, y_one_hot

    x_train, y_train = pre_process_data("../mnist-in-csv/mnist_train.csv")
    x_test, y_test = pre_process_data("../mnist-in-csv/mnist_test.csv")
    
    model = NeuralNetwork([LinearLayer((784, 100)), BatchNormLayer(100),
                           SigmoidLayer(),
                           LinearLayer((100, 100)), BatchNormLayer(100),
                           SigmoidLayer(),
                           LinearLayer((100, 100)), BatchNormLayer(100),
                           SigmoidLayer(),
                           LinearLayer((100, 10)),  SigmoidLayer()])
    model.fit(x_train, y_train, steps=50000, learning_rate=0.03, 
              batch_size=60, test_x=x_test, test_y=y_test)

