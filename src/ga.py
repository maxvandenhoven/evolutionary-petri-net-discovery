import itertools
import random

import numpy as np
from tqdm import tqdm
from deap import base, creator, tools, algorithms

from .eval import evaluate_individual


def run_petrinas_ga(
    dataset: list[list[int]],
    n_places: int,
    n_transitions: int,
    weight: float,
    crossover_ops: list[dict],
    mutation_ops: list[dict],
    selection_ops: list[dict],
    crossover_probs: list[float],
    mutation_probs: list[float],
    n_individuals: int,
    n_generations: int,
    n_iterations: int,
) -> list[dict]:
    """ Run GA optimization for petri net discovery problem with various hyperparameters

    Args:
        dataset (list[list[int]]): dataset of simulated traces
        n_places (int): number of places in the petri net
        n_transitions (int): number of transitions in the petri net
        weight (float): weight in [0, 1] on net validity versus sparseness
        crossover_ops (list[dict]): list of crossover operations to test. Each dictionary
            must contain a key "function" with the desired operation function. Parameters
            for the crossover function may be supplied as well
        mutation_ops (list[dict]): list of mutation operations to test. Each dictionary
            must contain a key "function" with the desired operation function. Parameters
            for the mutation function may be supplied as well
        selection_ops (list[dict]): list of selection operations to test. Each dictionary
            must contain a key "function" with the desired operation function. Parameters
            for the selection function may be supplied as well
        crossover_probs (list[float]): list of crossover probabilities to test
        mutation_probs (list[float]): list of mutation probabilities to test
        n_individuals (int): number of individuals in the population
        n_generations (int): number of generations until termination
        n_iterations (int): number of repetitions 

    Returns:
        list[dict]: list of metadata and optimization outputs for each run
    """       
    outputs = []

    # Create global DEAP container types for fitness and individuals
    creator.create("Fitness", base=base.Fitness, weights=(1.0,))
    creator.create("Individual", base=list, fitness=creator.Fitness)
    
    # Create all combinations of hyperparameters
    combinations = list(itertools.product(
        crossover_ops,
        mutation_ops,
        selection_ops,
        crossover_probs,
        mutation_probs,
    ))

    for combination in combinations:
        # Unpack combination
        crossover_op, mutation_op, selection_op, crossover_prob, mutation_prob = combination

        # Logging
        print("Crossover op:", crossover_op["function"].__name__)
        print("Mutation op:", mutation_op["function"].__name__)
        print("Selection op:", selection_op["function"].__name__)
        print("Crossover prob:", crossover_prob)
        print("Mutation prob:", mutation_prob)

        best_fitness = -np.infty
        sum_best_fitness = 0

        for iteration in tqdm(range(1, n_iterations + 1)):
            # Initialize output dictionary for run
            run_output = dict(
                iteration=iteration,
                crossover_op=crossover_op,
                mutation_op=mutation_op,
                selection_op=selection_op,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
            )

            # Create new toolbox and register functions to generate individuals
            # and populations
            toolbox = base.Toolbox()
            toolbox.register(
                alias="attr_edge", 
                function=random.randint, 
                a=-1, b=1
            )
            toolbox.register(
                alias="individual", 
                function=tools.initRepeat, 
                container=creator.Individual,
                func=toolbox.attr_edge,
                n=n_places*n_transitions,
            )
            toolbox.register(
                alias="population", 
                function=tools.initRepeat, 
                container=list, func=toolbox.individual
            )

            # Register evolutionary operators
            toolbox.register(alias="mate", **crossover_op)
            toolbox.register(alias="mutate", **mutation_op)
            toolbox.register(alias="select", **selection_op)
            toolbox.register(
                alias="evaluate",
                function=evaluate_individual,
                weight=weight,
                dataset=dataset,
                n_places=n_places,
                n_transitions=n_transitions,
            )

            # Create stat tracking objects
            halloffame = tools.HallOfFame(1)
            stats = tools.Statistics(lambda individual: individual.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            # Create population
            population = toolbox.population(n=n_individuals)

            # Run algorithm
            population, logbook = algorithms.eaSimple(
                population=population,
                toolbox=toolbox,
                stats=stats,
                halloffame=halloffame,
                cxpb=crossover_prob,
                mutpb=mutation_prob,
                ngen=n_generations,
                verbose=False,
            )

            # Compute best fitness in iteration and update statistics
            best_fitness_iteration = toolbox.evaluate(halloffame[0])[0]
            sum_best_fitness += best_fitness_iteration
            best_fitness = max(best_fitness, best_fitness_iteration)

            # Save results from run
            run_output["logbook"] = logbook
            run_output["halloffame"] = halloffame
            run_output["best_fitness"] = best_fitness_iteration
            outputs.append(run_output)

        print("Best fitness:", best_fitness)
        print("Average best fitness:", sum_best_fitness / n_iterations)
        print() # Better formatting

    return outputs

