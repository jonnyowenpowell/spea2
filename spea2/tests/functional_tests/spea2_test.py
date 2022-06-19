"""Run functional test of SPEA2 algorithm."""
from numba import jit
import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

from meal_plan_algorithm.spea2 import spea2

def run():
  chromosome_length = 10
  population_size = 100
  archive_size = 20
  crossover_proportion = 0.4

  x = np.linspace(1, 10, 10)
  y = np.zeros((20, 10)) + 50
  lines = []
  plt.ion()
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_ylim([-50, 150])

  for i in range(0, 20):
    line, = ax.plot(x, y[i], 'r-')
    lines.append(line)

  @jit(nopython=True, cache=True)
  def objectives(population):
    output = np.zeros((len(population), 3))
    for i in range(0, len(population)):
      chromosome = population[i]

      # All between -100 and 100
      all_bound_objective = np.int(np.all(np.greater(chromosome, -100))) + np.int(np.all(np.less(chromosome, 100)))

      # First between -50 and 50
      first_bound_objective = np.int(chromosome[0] < 50) + np.int(chromosome[0] > -50)

      # Sum to 300
      three_hundred_objective = -np.absolute(np.sum(chromosome) - 300)

      output[i, 0] = all_bound_objective
      output[i, 1] = first_bound_objective
      output[i, 2] = three_hundred_objective

    return output

  def generate_population():
    population = np.zeros((population_size, chromosome_length))
    i = 0
    while i < population_size:
      chromosome_candidate = (npr.rand(chromosome_length) - 0.5) * 45
      if not chromosome_candidate in population:
        population[i] = chromosome_candidate
        i += 1
    return population

  def termination_condition(archive, generation):
    scores = objectives(archive)[0]
    print(np.absolute(scores[2]))

    for i in range(0, 20):
      lines[i].set_ydata(archive[i])
      fig.canvas.draw()

    return (scores[0] == 2 and scores[0] == scores[1] and -scores[2] < 0.001)

  def crossover_chromosomes(parent_a, parent_b):
    point = npr.randint(chromosome_length)
    child_a = np.concatenate((parent_a[:point], parent_b[point:]))
    child_b = np.concatenate((parent_b[:point], parent_a[point:]))
    return (child_a, child_b)

  def mutate_chromosome(parent):
    return np.apply_along_axis(npr.normal, 0, parent, 10)

  results = spea2.run(population_size, archive_size, crossover_proportion, objectives, generate_population, termination_condition, crossover_chromosomes, mutate_chromosome)
  return (results, objectives)
