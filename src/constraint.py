import random


def flow_constraint(
    n_places: int, 
    n_transitions: int, 
    max_nonzero: int,
):
    """ Build decorator function to apply flow constraint to petri net solutions,
        limiting ingoing and outgoing connections to transitions

    Args:
        n_places (int): number of possible places in the net (rows)
        n_transitions (int): number of possible transitions in the net (columns)
        max_nonzero (int): maximum number of nonzero elements in a column
    """        
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)

            for individual in offspring:
                for col_index in range(n_transitions):
                    nonzero_indices = []
                    for row_index in range(n_places):
                        index = row_index*n_transitions + col_index
                        if individual[index] != 0:
                            nonzero_indices.append(index)
                    
                    if len(nonzero_indices) <= max_nonzero:
                        continue

                    random.shuffle(nonzero_indices)
                    modify_indices = nonzero_indices[max_nonzero:]

                    for index in modify_indices:
                        individual[index] = 0

            return offspring
        return wrapper
    return decorator








