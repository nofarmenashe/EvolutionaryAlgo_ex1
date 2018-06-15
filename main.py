import numpy as np
import sklearn.datasets
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

    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')

    data = list(zip(mnist.data, mnist.target))
    random.shuffle(data)
    data = [(x[0] / 255.0, transform_target(x[1])) for x in data]
    # data = [(x[0].astype(bool).astype(int), transform_target(x[1])) for x in data]

    train_data = data[:training]
    val_data = data[training:training + val]
    test_data = data[training + val:]

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
    hidden_layers_sizes = [200, 100]
    epochs = 30

    nn_args = BackpropArgs(input_size, output_size, learning_rate, hidden_layers_sizes, epochs)
    NNModel = BackPropModel(nn_args)

    population_size = 50
    replication_rate = 0.4
    mutation_rate = 0.7
    elitism_rate = 1

    GA_args = GAArgs(population_size, replication_rate, mutation_rate, elitism_rate, NNModel)
    print(GA_args.population_size, GA_args.replication_rate, GA_args.mutation_rate, GA_args.elitism_rate)

    GA = GAModel(GA_args)

    random.shuffle(train_data)
    # random.shuffle(test_data)
    GA.train(train_data, val_data, test_data)
