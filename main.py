import numpy as np
import mnist
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

    mnist_data = mnist.MNIST('./data')

    train_x, train_y = mnist_data.load_training()
    test_x, test_y = mnist_data.load_testing()
    train_val_data = [(np.array(x) / 255.0, transform_target(y)) for x, y in zip(train_x, train_y)]
    test_data = [(np.array(x) / 255.0, transform_target(y)) for x, y in zip(test_x, test_y)]
    random.shuffle(train_val_data)

    train_data = train_val_data[:training]
    val_data = train_val_data[training:]

    return train_data, val_data, test_data


if __name__ == "__main__":
    print("loading dataset")
    train_data, val_data, test_data = load_datasets()
    input_size = 28 * 28
    output_size = 10

    # part = sys.argv[1]

    # if part == 'a':
    #     print("start backprop")
    #     backprop_args = BackpropArgs(28*28, 10, 0.01, [240, 120], 30)
    #     print(backprop_args.learning_rate)
    #     print(backprop_args.hidden_layers_sizes)
    #     backProp = BackPropModel(backprop_args)
    #
    #
    #     backProp.train(train_data, val_data)
    #     print("Test Accuracy:", str(backProp.test(test_data)) + "%")
    #     print("Train Accuracy:", str(backProp.test(train_data)) + "%")

    # if part == 'b':
    print("start GA")
    learning_rate = 0.01
    hidden_layers_sizes = [128, 64]
    epochs = 1

    nn_args = BackpropArgs(input_size, output_size, learning_rate, hidden_layers_sizes, epochs)
    NNModel = BackPropModel(nn_args)

    population_size = 100
    replication_rate = 0.1
    mutation_rate = 0.1
    elitism_rate = 2

    GA_args = GAArgs(population_size, replication_rate, mutation_rate, elitism_rate, NNModel)
    print(GA_args.population_size, GA_args.replication_rate, GA_args.mutation_rate, GA_args.elitism_rate)

    GA = GAModel(GA_args)

    random.shuffle(train_data)

    GA.train(train_data, val_data, test_data)
