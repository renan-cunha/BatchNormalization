from src.feed_forward.neural_network import NeuralNetwork
from src.feed_forward.layers import LinearLayer, SigmoidLayer, BatchNormLayer
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


learning_rate = 0.01
batch_size = 60
neurons = 100
steps = 11000

def pre_process_data(location: str) -> Tuple[np.ndarray, np.ndarray]:
    csv_train = pd.read_csv(location)
    X = csv_train.values[:, 1:].astype("float32")
    X /= 255
    y = csv_train.values[:, 0]
    y_one_hot = np.zeros((y.size, y.max()+1))
    y_one_hot[np.arange(y.size), y] = 1
    y_one_hot = y_one_hot.astype("float32")
    return X, y_one_hot


x_train, y_train = pre_process_data("src/mnist-in-csv/mnist_train.csv")
x_test, y_test = pre_process_data("src/mnist-in-csv/mnist_test.csv")

model_bn = NeuralNetwork([LinearLayer((x_train.shape[1], neurons)), 
                          BatchNormLayer(neurons),
                          SigmoidLayer(),
                          LinearLayer((neurons, neurons)), 
                          BatchNormLayer(neurons),
                          SigmoidLayer(),
                          LinearLayer((neurons, neurons)), 
                          BatchNormLayer(neurons),
                          SigmoidLayer(),
                          LinearLayer((neurons, y_train.shape[1])),  
                          SigmoidLayer()])

model_baseline = NeuralNetwork([LinearLayer((x_train.shape[1], neurons)), 
                                SigmoidLayer(),
                                LinearLayer((neurons, neurons)), 
                                SigmoidLayer(),
                                LinearLayer((neurons, neurons)), 
                                SigmoidLayer(),
                                LinearLayer((neurons, y_train.shape[1])),
                                SigmoidLayer()])

baseline_accuracy = model_baseline.fit(x_train, y_train, steps=steps, 
                                       learning_rate=learning_rate, 
                                       batch_size=batch_size, 
                                       test_x=x_test, test_y=y_test)[1]
bn_accuracy = model_bn.fit(x_train, y_train, steps=steps, 
                           learning_rate=learning_rate, 
                           batch_size=batch_size, test_x=x_test, 
                           test_y=y_test)[1]

print(baseline_accuracy.shape)
plt.plot(baseline_accuracy, label="Baseline")
plt.plot(bn_accuracy, label="BatchNorm")
plt.show()

