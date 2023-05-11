def load_dataset(filepath: str, transitions: list[str]) -> list[list[int]]:
    """ Load dataset of traces

    Args:
        filepath (str): path to .txtg file containing traces on each line
        transitions (list[str]): list of all transitions in the model

    Returns:
        list[list[int]]: list of traces, where each trace is a list of integers 
            representing transitions
    """    
    traces = []
    with open(filepath, "r") as file:
        for line in file:
            trace = line.strip().split(", ")
            # Convert transitions in trace to integers
            trace = [transitions.index(transition) for transition in trace]
            traces.append(trace)

    return traces