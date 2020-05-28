from src.feed_forward.neural_network import NeuralNetwork
from src.feed_forward.layers import LinearLayer, SigmoidLayer, BatchNormLayer
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


learning_rate = 0.0005
batch_size = 60
neurons = 100
steps = 51000
mnist_dir = "data/"


def pre_process_data(location: str) -> Tuple[np.ndarray, np.ndarray]:
    csv_train = pd.read_csv(location)
    X = csv_train.values[:, 1:].astype("float32")
    X /= 255
    y = csv_train.values[:, 0]
    y_one_hot = np.zeros((y.size, y.max()+1)).astype("float32")
    y_one_hot[np.arange(y.size), y] = 1
    return X, y_one_hot


x_train, y_train = pre_process_data(f"{mnist_dir}/mnist_train.csv")
x_test, y_test = pre_process_data(f"{mnist_dir}/mnist_test.csv")

model_bn = NeuralNetwork([LinearLayer(x_train.shape[1], neurons),
                          BatchNormLayer(neurons),
                          SigmoidLayer(),
                          LinearLayer(neurons, neurons),
                          BatchNormLayer(neurons),
                          SigmoidLayer(),
                          LinearLayer(neurons, neurons),
                          BatchNormLayer(neurons),
                          SigmoidLayer(),
                          LinearLayer(neurons, y_train.shape[1]),
                          SigmoidLayer()])

model_baseline = NeuralNetwork([LinearLayer(x_train.shape[1], neurons),
                                SigmoidLayer(),
                                LinearLayer(neurons, neurons),
                                SigmoidLayer(),
                                LinearLayer(neurons, neurons),
                                SigmoidLayer(),
                                LinearLayer(neurons, y_train.shape[1]),
                                SigmoidLayer()])
print("Training Baseline")
baseline_accuracy = model_baseline.fit(x_train, y_train, steps=steps,
                                       learning_rate=learning_rate,
                                       batch_size=batch_size,
                                       x_test=x_test, y_test=y_test)[:, 1]
print("Training Model w/ Batch Norm")
bn_accuracy = model_bn.fit(x_train, y_train, steps=steps,
                           learning_rate=learning_rate,
                           batch_size=batch_size, x_test=x_test,
                           y_test=y_test)[:, 1]

num_batches = math.ceil(x_train.shape[0] / batch_size)
steps_axis = np.linspace(1, steps/num_batches, len(baseline_accuracy)//10+1)
steps_axis = steps_axis.astype("int")

plt.plot(baseline_accuracy, label="Baseline")
plt.plot(bn_accuracy, label="Model w/ BatchNorm")
plt.ylabel("Test Accuracy")
plt.xlabel("Steps")
plt.title("Results on MNIST")
locs = np.linspace(0, len(baseline_accuracy)-1, len(baseline_accuracy)//10+1)
plt.xticks(locs, [str(x)+"k" for x in steps_axis])
plt.grid()
plt.legend()
plt.show()