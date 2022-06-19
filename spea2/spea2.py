# Module to run SPEA2 algorithm
import numpy as np

from spea2 import binary_tournament, densities, distance_matrix, domination_scores


def run(
    population_size,
    archive_size,
    crossover_proportion,
    objectives,
    generate_population,
    termination_condition,
    crossover_chromosomes,
    mutate_chromosome,
):
    # Generate initial random population
    population = generate_population()

    # Create empty archive
    archive = np.zeros((0, population.shape[1]), dtype=population.dtype)

    archive_k = int(min(np.round(np.sqrt(archive_size)), archive_size - 1))

    # Set generation to 0
    generation = 0

    # Generational loop
    while True:
        # Create multiset union of previous generation population and archive
        population_archive_union = population
        for i in range(len(archive)):
            if not archive[i] in population:
                population = np.append(population, [archive[i]], axis=0)

        # Calculate k for size of union
        k = int(np.round(np.sqrt(len(population_archive_union))))

        # Evaluate fitness of individuals in the multiset
        union_objective_scores = objectives(population_archive_union)
        union_domination_scores = domination_scores.calculate(union_objective_scores)
        union_distance_matrix = distance_matrix.calculate(population_archive_union)
        union_densities = densities.calculate(union_distance_matrix, k)
        union_fitnesses = np.add(union_domination_scores, union_densities)

        # Find non dominated set (current approximate Pareto front)
        non_dominated_mask = union_fitnesses < 1
        non_dominated = population_archive_union[non_dominated_mask]
        non_dominated_size = len(non_dominated)

        # Fill archive with non dominated individuals, adjusting for size via truncation or adding dominated individuals
        cardinality_difference = archive_size - non_dominated_size
        if cardinality_difference == 0:
            archive = non_dominated
        elif cardinality_difference > 0:
            indices = union_fitnesses.argsort()
            sorted_union = population_archive_union[indices]
            archive = np.concatenate(
                (
                    non_dominated,
                    sorted_union[
                        non_dominated_size : non_dominated_size + cardinality_difference
                    ],
                )
            )
        else:
            for i in range(-cardinality_difference):
                neighbour_distance = 1
                maximum_neighbour_distance = len(non_dominated) - 1
                non_dominated_distance_matrix = distance_matrix.calculate(non_dominated)
                non_dominated_distance_matrix.sort()
                while (
                    max(non_dominated_distance_matrix[neighbour_distance, :])
                    == min(non_dominated_distance_matrix[neighbour_distance, :])
                    and neighbour_distance < maximum_neighbour_distance
                ):
                    neighbour_distance += 1
                print(neighbour_distance)
                minimum_distance_neighbour = min(
                    (distance, index)
                    for (index, distance) in enumerate(
                        non_dominated_distance_matrix[neighbour_distance, :]
                    )
                )[1]
                non_dominated = np.delete(
                    non_dominated, minimum_distance_neighbour, axis=0
                )
            archive = non_dominated

        # Break out of main loop if the maximum number of generations is reached
        if termination_condition(archive, generation):
            return archive
        generation += 1

        # Find distance matrix for archive
        archive_distance_matrix = distance_matrix.calculate(archive)
        archive_densities = densities.calculate(archive_distance_matrix, archive_k)

        # Create section of new population from crossover
        n_crossovers = int(np.round(population_size * crossover_proportion * 0.5))
        crossovers = np.zeros(
            (n_crossovers * 2, population.shape[1]), dtype=population.dtype
        )
        for i in range(n_crossovers):
            parent_a = binary_tournament.perform(archive_densities)
            parent_b = binary_tournament.perform(archive_densities)
            crossovers[i * 2], crossovers[i * 2 + 1] = crossover_chromosomes(
                archive[parent_a], archive[parent_b]
            )

        # Create section of new population from mutation
        n_mutations = int(population_size - n_crossovers * 2)
        mutations = np.zeros((n_mutations, population.shape[1]), dtype=population.dtype)
        for i in range(n_mutations):
            parent = binary_tournament.perform(archive_densities)
            mutations[i] = mutate_chromosome(archive[parent])

        # Create next generation population from crossedover and mutated individuals
        population = np.concatenate((crossovers, mutations))
