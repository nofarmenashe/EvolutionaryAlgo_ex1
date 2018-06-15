import numpy as np
from mnist import MNIST
import sys
from numpy import random
from backprop_algorithm import BackpropArgs, BackPropModel
from genetic_algorithm import GAArgs, GAModel


def transform_target(y):
    t = np.zeros((10, 1))
    t[int(y)] = 1.0
    return t


def load_datasets():
    training = 50000
    val = 10000

    mnist_data = MNIST('samples')

    train_val_data = mnist_data.load_training()
    test_data = mnist_data.load_testing()

    random.shuffle(train_val_data)
    train_val_data = [(x[0] / 255.0, transform_target(x[1])) for x in train_val_data]
    test_data = [(x[0] / 255.0, transform_target(x[1])) for x in test_data]

    train_data = train_val_data[:training]
    val_data = train_val_data[training:]
    print(train_data.shape, val_data.shape, test_data.shape, train_data[0])

    return train_data, val_data, test_data


if __name__ == "__main__":
    print("loading dataset")
    train_data, val_data, test_data = load_datasets()

    part = sys.argv[1]

    if part == 'a':
        print("start backprop")
        backprop_args = BackpropArgs(28*28, 10, 0.01, [240, 120], 30)
        print(backprop_args.learning_rate)
        print(backprop_args.hidden_layers_sizes)
        backProp = BackPropModel(backprop_args)

        backProp.train(train_data, val_data)
        print("Test Accuracy:", str(backProp.test(test_data)) + "%")
        print("Train Accuracy:", str(backProp.test(train_data)) + "%")

    if part == 'b':
        print("start GA")
        nn_args = BackpropArgs(28 * 28, 10, 0.01, [240, 120], 30)
        NNModel = BackPropModel(nn_args)
        GA_args = GAArgs(30, 0.1, 0.2, 0.1, NNModel)
        print(GA_args.population_size, GA_args.mutation_rate, GA_args.replication_rate, GA_args.elitism_rate)
        GA = GAModel(GA_args)
        GA.train(train_data, val_data, test_data)
