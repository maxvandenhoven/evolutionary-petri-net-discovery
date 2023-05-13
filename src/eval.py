import numpy as np


def evaluate_individual(
    individual: list[int], 
    weight: float,
    dataset: list[list[int]],
    n_places: int,
    n_transitions: int,
) -> tuple[int, int]: 
    """ Evaluate flattened matrix on daatset of simulated traces

    Args:
        individual (list[int]): list of petri net edge weights
        weight (float): weight in [0, 1]
        dataset (list[list[int]]): collection of simulated traces
        n_places (int): number of places in the petri net
        n_transitions (int): number of transitions in the petri net

    Returns:
        tuple[int, int]: number of successful trace steps and number of nonzero edges
    """    
    # Convert flattened individual to matrix form
    individual_matrix = np.array(individual).reshape(n_places, n_transitions)

    counter = 0

    for trace in dataset:
        counter += evaluate_trace(individual_matrix, trace)

    return weight*counter - (1-weight)*np.count_nonzero(individual_matrix),


def evaluate_trace(individual_matrix: np.array, trace: list[int]) -> int:
    """ Evaluate matrix-form of individual on a single trace

    Args:
        individual_matrix (np.array): matrix of shape (n_places, n_transitions)
        trace (list[int]): list of transitions in a simulated trace

    Returns:
        int: number of transitions in simulated trace supported by the matrix
    """  
    # Initialize place and counter to 0 (start at first place)
    place = 0  
    counter = 0

    for transition in trace:
        # Check if current place has a connection to the next transition in the trace. If
        # yes, increment the counter and determine the next place. If not, break the loop
        # and return the counter
        if individual_matrix[place, transition] == -1:
            counter += 1

            # Loop over the column corresponding to the current transition in the trace.
            # If a place has a connection to the transition, update the current place to 
            # that new place and stop looking. 
            for candidate_place, edge in enumerate(individual_matrix[:, transition]):
                if edge == 1:
                    place = candidate_place
                    break
        else:
            break

    return counter