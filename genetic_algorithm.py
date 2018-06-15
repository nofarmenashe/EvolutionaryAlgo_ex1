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
        self.rouletteProbs = self.init_roulette_probs()

    def init_roulette_probs(self):
        sum = np.sum(range(1, self.population_size + 1))
        fitnesses = [float(i) / sum for i in range(1, self.population_size + 1)]
        fitnesses = fitnesses[::-1]
        return fitnesses

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

        loss, success = new_nn.calculate_loss_and_success(train_dataset)
        return nn_chromosome, loss, success

        # weights, biases = nn_chromosome
        # new_nn = BackPropModel(self.nn.args)
        # new_nn.weights = list(weights)
        # new_nn.biases = list(biases)
        # accuracy = new_nn.test(train_dataset)
        # # print(str(accuracy) + "%")
        # return accuracy

    def replication(self, population_list):
        top_permutaions = population_list[:int(self.replication_rate * len(population_list))]
        return [fitnessed_permutation[0] for fitnessed_permutation in top_permutaions]
        # replicated_chromosomes = random.sample(population_list, k=int(self.replication_rate * self.population_size))
        # return replicated_chromosomes

    def choose_parents(self, population_fitness_tuples):
        networks = [nn_loss_accuracy[0] for nn_loss_accuracy in population_fitness_tuples]
        networks = list(networks)

        chosen_indeces = np.random.choice(range(len(networks)), 2, p=self.rouletteProbs)
        return networks[chosen_indeces[0]], networks[chosen_indeces[1]]

    def breed_parents(self, parent1, parent2):
        parent1_weights, parent1_biases = parent1
        parent2_weights, parent2_biases = parent2
        child_weights = []
        child_biases = []
        for w1, w2 in zip(parent1_weights, parent2_weights):
            new_w = np.zeros(w1.shape)
            for i in range(w1.shape[1]):
                new_w[:, i] += random.choice([w1[:, i], w2[:, i]])

            child_weights.append(new_w)
        # for parent1_weight, w2 in zip(parent1_weights, parent2_weights):
        #     new_w = np.zeros(parent1_weight.shape)
        #     for i in range(len(parent1_weight)):
        #         new_w[i] = random.choice([parent1_weight[i], w2[i]])
        #     child_weights.append(new_w)

        for b1, b2 in zip(parent1_biases, parent2_biases):
            new_b = np.zeros(b1.shape)
            for i in range(b1.shape[1]):
                new_b += random.choice([b1, b2])
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
        chromosome_w, chromosome_b = chromosome
        for w in chromosome_w:
            w += np.random.normal(loc=0.0, scale=0.05, size=w.shape)
            new_w.append(w)

        for b in chromosome_b:
            b += np.random.normal(loc=0.0, scale=0.05, size=b.shape)
            new_b.append(b)

        return new_w, new_b

    def population_mutation(self, population):
        mutated_population = []
        for chromosome in population:
            if calculate_probability(self.mutation_rate):
                mutated_population.append(self.mutate(chromosome))
            else:
                mutated_population.append(chromosome)
        return mutated_population

    def train(self, train_dataset, val_dataset, test_dataset):
        best_fitness = (None, 0)
        generation_number = 1
        small_train_set = train_dataset[:100]
        while best_fitness[1] < 98:
            nn_and_fitness = []
            new_population = []

            # train_batch = random.sample(train_dataset, k=100)
            # random.shuffle(train_dataset)
            sample_trainset = random.sample(train_dataset, 200)

            # calculate fitnesses
            # chromosome = (nn, loss, accuracy)
            nn_and_fitness.extend([self.fitness(nn, sample_trainset) for nn in self.population])

            nn_and_fitness.sort(key=operator.itemgetter(2))
            nn_and_fitness = nn_and_fitness[::-1]
            print("Accuracy: ", [format(p[2], '.2f') for p in nn_and_fitness])

            nn_and_fitness.sort(key=operator.itemgetter(1))
            print("loss: ", [format(p[1], '.2f') for p in nn_and_fitness])
            best_fitness = nn_and_fitness[0]

            elit_chromosomes = [copy.deepcopy(nn_and_fitness[0]) for nn_and_fitness
                                in nn_and_fitness[:self.elitism_rate]]

            # replication - select randomly from the rest
            rest_of_population = [population_fitness[0] for population_fitness
                                  in nn_and_fitness[int(self.elitism_rate * self.population_size):]]
            replications = self.replication(rest_of_population)
            new_population.extend(replications)

            # crossover - breed random parents
            num_of_chromosomes_left = self.population_size - len(new_population) - self.elitism_rate
            crossover_children = self.crossover(nn_and_fitness, num_of_chromosomes_left)
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