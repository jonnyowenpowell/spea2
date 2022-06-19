"""Find euclidean distance matrix for a population."""
import scipy.spatial.distance


def calculate(population):
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(population))


"""
Equivalent code without scipy requirement.
"""
"""
def calculate(population):
  size = len(population)
  distance_matrix = np.zeros((size, size))

  for i in range(size):
    for j in range(i + 1, size):
      distance = np.sum((population[i] - population[j])**2)**0.5
      distance_matrix[i, j] = distance
      distance_matrix[j, i] = distance

  return distance_matrix
"""
