import numpy as np
from typing import Tuple, List
from src.feed_forward.layers import CrossEntropyLoss, ParamLayer, Layer, BatchNormLayer
from tqdm import tqdm
import math


class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers[:]
        self.loss_layer: Layer
    
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
                       test_y: np.ndarray,
                       x_train: np.ndarray) -> Tuple[float, float]:
        """Returns loss and accuracy in the test set"""
        self.loss_layer = CrossEntropyLoss(test_y)
        y_pred = self.predict(test_x, x_train=x_train)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_labels = np.argmax(test_y, axis=1)
        loss = self.loss_layer.forward(y_pred)
        accuracy = np.sum(y_labels == y_pred_labels) / test_x.shape[0]
        return float(np.mean(loss)), accuracy

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            learning_rate: float, steps: int,
            batch_size: int, x_test: np.ndarray,
            y_test: np.ndarray) -> np.array:
        
        num_examples = x_train.shape[0]
        num_batches = math.ceil(num_examples/batch_size)
        metrics = np.zeros((steps//num_batches, 2))  # loss, accuracy
        random_index = np.linspace(0, num_examples-1, num_examples).astype(int)
        for epoch in tqdm(range(steps//num_batches)):
            np.random.shuffle(random_index)
            x_train = x_train[random_index]
            y_train = y_train[random_index]
            for index_batch in range(0, num_examples, batch_size):
                mini_batch_x = x_train[index_batch: index_batch + batch_size]
                mini_batch_y = y_train[index_batch: index_batch + batch_size]
                self.loss_layer = CrossEntropyLoss(mini_batch_y)
                y_pred = self.predict(mini_batch_x, test=False)
                self.loss_layer.forward(y_pred)
                self.backward_propagation(y_pred)
                self.update_params(learning_rate)

            loss, accuracy = self.evaluate_model(x_test, y_test, x_train)
            metrics[epoch, :] = loss, accuracy
            print(f"Epoch {epoch}")
            print(f"Loss: {loss} | Accuracy: {accuracy}")                   

        return metrics

    def predict(self, x_test: np.ndarray, test: bool = True,
                x_train:np.ndarray = None) -> np.ndarray:
        if test:
            for layer in self.layers:
                if type(layer) == BatchNormLayer:
                    layer.set_variables(x_train)
                x_train = layer.forward(x_train, predict=test)

        for layer in self.layers:
            x_test = layer.forward(x_test, predict=test)
        return x_test


