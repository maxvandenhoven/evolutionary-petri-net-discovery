import json
from copy import deepcopy

import numpy as np


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


def save_graph(
    individual: list,
    n_places: int,
    n_transitions: int,
    transitions: list[str],
    output_file: str,
):
    from pm4py.objects.petri_net.obj import PetriNet
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

    # Convert flattened individual to matrix form
    individual_matrix = np.array(individual).reshape(n_places, n_transitions)

    net = PetriNet("PETRINAS")

    for d in range(n_places):
        p = PetriNet.Place("p%d" % (d + 1))
        net.places.add(p)

    for d in range(n_transitions):
        t = PetriNet.Transition(transitions[d], transitions[d])
        net.transitions.add(t)

    for t, transition in enumerate(individual_matrix.T):
        for p, place in enumerate(transition):

            # ingoing
            if place == 1:

                from_node1 = to_node1 = None

                for from_tran in list(net.transitions):
                    if str(from_tran).split(", ")[0][1:] != transitions[t]: continue
                    from_node1 = from_tran

                for to_place in net.places:
                    if str(to_place) != 'p%d' % (p + 1): continue
                    to_node1 = to_place

                add_arc_from_to(from_node1, to_node1, net)

            elif place == -1:

                from_node2 = to_node2 = None

                for from_place in net.places:
                    if str(from_place) != 'p%d' % (p + 1): continue
                    from_node2 = from_place

                for to_tran in list(net.transitions):
                    if str(to_tran).split(", ")[0][1:] != transitions[t]: continue
                    to_node2 = to_tran

                add_arc_from_to(from_node2, to_node2, net)

    parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "pdf"}
    gviz = pn_visualizer.apply(net, parameters=parameters)
    pn_visualizer.save(gviz, output_file)