# Module to contain hypothesis helper functions
from hypothesis.strategies import tuples, integers, floats
from hypothesis.extra.numpy import arrays

default_elements = floats(allow_nan=False, allow_infinity=False)

def rnd_len_1d_arrays(dtype, min_len, max_len, elements=default_elements):
  lengths = integers(min_value=min_len, max_value=max_len)
  return lengths.flatmap(lambda l: arrays(dtype, l, elements=elements))

def rnd_len_1d_array_pairs(dtype, min_len, max_len, elements=default_elements):
  lengths = integers(min_value=min_len, max_value=max_len)
  return lengths.flatmap(lambda l: tuples(arrays(dtype, l, elements=elements), arrays(dtype, l, elements=elements)))

def rnd_pairwise_len_1d_arrays(dtype, min_pairwise_number_len, max_pairwise_number_len, elements=default_elements):
  pairwise_numbers = integers(min_value=min_pairwise_number_len, max_value=max_pairwise_number_len)
  return pairwise_numbers.flatmap(lambda p_n: arrays(dtype, int((p_n**2-p_n)/2), elements=elements))

def rnd_nd_dims(dims, min_len, max_len):
  dimension_strategies = []
  for n in range(dims):
    dimension_strategies.append(integers(min_value=min_len, max_value=max_len))
  return tuples(*dimension_strategies)