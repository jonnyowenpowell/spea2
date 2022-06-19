"""Find domination scores of population."""
import spea2dominationscores


def calculate(objective_scores):
    return spea2dominationscores.domination_scores(objective_scores)


"""
Equivalent code without C module requirement.
"""
"""
def calculate(objective_scores):
  size = len(objective_scores)
  domination_matrix = np.zeros((size, size))
  strengths = np.zeros((size))
  domination_scores = np.zeros((size))

  for i in range(size):
    for j in range(i+1, size):
      a = objective_scores[i]
      b = objective_scores[j]
      if np.all(a >= b) and np.any(a > b):
        strengths[i] += 1
        domination_matrix[j, i] = 1
      elif np.all(b >= a) and np.any(b > a):
        strengths[j] += 1
        domination_matrix[i, j] = 1

  for i in range(size):
    for j in range(size):
      if domination_matrix[i, j]:
        domination_scores[i] += strengths[j]

  return domination_scores
  """
