import operator
import random
import numpy as np
import copy

from backprop_algorithm import BackPropModel, BackpropArgs


def calculate_probability(p):
    return random.random() >= 1-p


class GAArgs:
    def __init__(self, population_size, replication_rate, mutation_rate, elitism_rate, nn: BackPropModel):
        self.population_size = population_size
        self.replication_rate = replication_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.nn = nn


class GAModel:
    def __init__(self, args: GAArgs):
        self.population_size = args.population_size
        self.replication_rate = args.replication_rate
        self.mutation_rate = args.mutation_rate
        self.elitism_rate = args.elitism_rate
        self.nn = args.nn
        self.population = self.init_population()

    def init_population(self):
        population = []
        biases = np.array([np.zeros((y, 1)) for y in self.nn.layers[1:]])
        for i in range(self.population_size):
            weights = np.array([np.random.normal(loc=0.0, scale=0.1, size=(y, x))
                                for x, y in list(zip(self.nn.layers[:-1], self.nn.layers[1:]))])
            population.append((weights, biases))
        return population

    def fitness(self, nn_chromosome, train_dataset):
        weights, biases = nn_chromosome
        new_nn = BackPropModel(self.nn.args)
        new_nn.weights = list(weights)
        new_nn.biases = list(biases)
        accuracy = new_nn.test(train_dataset)
        # print(str(accuracy) + "%")
        return accuracy

    def calculate_accuracy(self, nn_chromosome, train_dataset):

    def replication(self, population_list):
        top_permutaions = population_list[:int(self.replication_rate * len(population_list))]
        return [fitnessed_permutation[0] for fitnessed_permutation in top_permutaions]
        # replicated_chromosomes = random.sample(population_list, k=int(self.replication_rate * self.population_size))
        # return replicated_chromosomes

    def choose_parents(self, population_fitness_tuples):
        networks, fitnesses = zip(*population_fitness_tuples)
        networks = list(networks)
        fitnesses = list(fitnesses)

        sum_fitnesses = np.sum(fitnesses)
        fitnesses = [float(fitness) / sum_fitnesses for fitness in fitnesses]
        chosen_indeces = np.random.choice(range(len(networks)), 2, p=fitnesses)
        return networks[chosen_indeces[0]], networks[chosen_indeces[1]]

    def breed_parents(self, parent1, parent2):
        parent1_weights, parent1_biases = parent1
        parent2_weights, parent2_biases = parent2

        parent1_weights, parent1_biases, parent2_weights, parent2_biases = [list(x) for x in [parent1_weights, parent1_biases, parent2_weights, parent2_biases]]

        child_weights = []
        child_biases = []

        for parent1_weight, parent2_weight in zip(parent1_weights, parent2_weights):
            new_w = np.zeros(parent1_weight.shape)
            for i in range(len(parent1_weight)):
                new_w[i] = random.choice([parent1_weight[i], parent2_weight[i]])
            child_weights.append(new_w)
        # for parent1_weight, w2 in zip(parent1_weights, parent2_weights):
        #     new_w = np.zeros(parent1_weight.shape)
        #     for i in range(len(parent1_weight)):
        #         new_w[i] = random.choice([parent1_weight[i], w2[i]])
        #     child_weights.append(new_w)

        for b1, b2 in zip(parent1_biases, parent2_biases):
            new_b = np.zeros(b1.shape)
            for i in range(len(b1)):
                new_b[i] = random.choice([b1[i], b2[i]])
            child_biases.append(new_b)

        return child_weights, child_biases

    def crossover(self, population_fitness_tuples, num_of_crossovers):
        children = []
        for i in range(num_of_crossovers):
            p1, p2 = self.choose_parents(population_fitness_tuples)
            children.append(self.breed_parents(p1, p2))
        return children

    def mutate(self, chromosome):
        new_w = []
        new_b = []
        chromosome_weights, chromosome_biases = chromosome

        # randon_normalized_biases = np.array([np.random.normal(loc=0.0, scale=1, size=(y, 1)) for y in self.nn.layers[1:]])
        # randon_normalized_weights = np.array([np.random.normal(loc=0.0, scale=1, size=(y, x))
        #                     for x, y in list(zip(self.nn.layers[:-1], self.nn.layers[1:]))])
        #
        # for w, normalized_w in zip(chromosome_w, randon_normalized_weights):
        #     if calculate_probability(self.mutation_rate):
        #         w = w + normalized_w
        #
        #     new_w.append(w)
            # mask = np.random.choice([0, 1], p=[1-self.mutation_rate, self.mutation_rate], size=w.shape).astype(np.bool)
            # values = w + np.random.normal(loc=0.0, scale=0.1, size=w.shape)
            # np.place(w, mask, values)

        for weight in chromosome_weights:
            w = np.zeros(weight.shape)
            for i, row in enumerate(weight):
                if calculate_probability(self.mutation_rate):
                    random_index = random.randint(0, len(row) - 1)
                    row[random_index] = 0
                w[i] = row

            new_w.append(w)

        # for b, normalized_b in zip(chromosome_b, randon_normalized_biases):
        #     if calculate_probability(self.mutation_rate):
        #         b = b + normalized_b
        #     # mask = np.random.choice([0, 1], p=[1 - self.mutation_rate, self.mutation_rate], size=b.shape).astype(np.bool)
        #     # values = b + np.random.normal(loc=0.0, scale=0.1, size=b.shape)
        #     # np.place(b, mask, values)
        #     new_b.append(b)

        for bias in chromosome_biases:
            b = np.zeros(bias.shape)
            for i, row in enumerate(bias):
                if calculate_probability(self.mutation_rate):
                    random_index = random.randint(0, len(row) - 1)
                    row[random_index] = 0
                b[i] = row
            new_b.append(b)

        return new_w, new_b

    def population_mutation(self, population):
        mutated_population = []
        for chromosome in population:
            mutated_population.append(self.mutate(chromosome))
        return mutated_population

    def train(self, train_dataset, val_dataset, test_dataset):
        best_fitness = (None, 0)
        generation_number = 1
        # train_set = train_dataset[:100]
        while best_fitness[1] < 98:
            population_fitnesses = []
            new_population = []

            # calculate fitnesses
            train_set = random.sample(train_dataset, 100)
            population_fitnesses.extend([(nn, self.fitness(nn, train_set))
                                         for nn in self.population])

            population_fitnesses.sort(key=operator.itemgetter(1))
            population_fitnesses = population_fitnesses[::-1]

            best_fitness = population_fitnesses[0]

            print([p[1] for p in population_fitnesses])

            elit_chromosomes = [copy.deepcopy(population_fitness[0]) for population_fitness
                                in population_fitnesses[:self.elitism_rate]]

            replications = self.replication(population_fitnesses)
            new_population.extend(replications)

            # crossover - breed random parents
            num_of_chromosomes_left = self.population_size - len(new_population) - self.elitism_rate
            crossover_children = self.crossover(population_fitnesses, num_of_chromosomes_left)
            new_population.extend(crossover_children)

            # mutation - mutate new population
            new_population = self.population_mutation(new_population)

            # elitism - select top
            new_population.extend(elit_chromosomes)

            print("Finished Generation: ", generation_number)
            generation_number += 1
            self.population = new_population

        accuracy = best_fitness[0].test(test_dataset)
        print("Test Acuuracy: " + str(accuracy))