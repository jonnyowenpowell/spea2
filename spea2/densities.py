"""Find density values for a population."""


def calculate(distances, k):
    distances.sort()
    kth_distances = distances[k, :]
    densities = 1 / (kth_distances + 2)
    return densities
