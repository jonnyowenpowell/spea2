"""Select a chromosome from a population by binary tournament."""
import numpy.random as npr


def perform(fitnesses):
    maximum_index = len(fitnesses) - 1

    index_a = npr.randint(maximum_index)
    index_b = npr.randint(maximum_index)

    fitness_a = fitnesses[index_a]
    fitness_b = fitnesses[index_b]

    if fitness_a < fitness_b:
        return index_a
    else:
        return index_b
