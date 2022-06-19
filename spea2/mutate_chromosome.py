# Module to mutate a chromosome
import numpy as np
import numpy.random as npr


def perform(parent, chromosome_length, sigma):
    return np.add(parent, np.multiply(sigma, npr.normal(size=chromosome_length))).clip(
        min=0
    )
