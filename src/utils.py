import json
from copy import deepcopy


def save_outputs(outputs: list[dict], filename: str):
    """ Save optimization outputs for later reference. Note that logbook object is 
        serialized as list of dictionaries

    Args:
        outputs (list[dict]): List of dictionaries to save
        filename (str): File to save to
    """    
    outputs_copy = deepcopy(outputs)

    for output in outputs_copy:
        # Serialize non-serializable objects manually
        try:
            output["crossover_op"]["function"] = output["crossover_op"]["function"].__name__
        except AttributeError:
            output["crossover_op"]["function"] = str(output["crossover_op"]["function"])

        try:
            output["mutation_op"]["function"] = output["mutation_op"]["function"].__name__
        except AttributeError:
            output["crossover_op"]["function"] = str(output["crossover_op"]["function"])

        try:
            output["selection_op"]["function"] = output["selection_op"]["function"].__name__
        except AttributeError:
            output["crossover_op"]["function"] = str(output["crossover_op"]["function"])

        output["halloffame"] = list(output["halloffame"])

    # Save to JSON
    with open(filename, "w+") as file:
        json.dump(outputs_copy, file)