# Module to crossover pairs of chromosomes
import numpy as np
import numpy.random as npr


def perform(parent_a, parent_b, chromosome_length, gamma):
    alpha = npr.uniform(low=(-1.0 * gamma), high=(gamma + 1), size=chromosome_length)
    child_a = np.add(np.multiply(alpha, parent_a), np.multiply((1 - alpha), parent_b))
    child_b = np.add(np.multiply(alpha, parent_b), np.multiply((1 - alpha), parent_a))
    return (child_a, child_b)
