##########################################################################################
# Imports
##########################################################################################
from deap import tools

from src.data import load_dataset
from src.ga import run_petrinas_ga


##########################################################################################
# Configuration
##########################################################################################
N_PLACES = 12
N_TRANSITIONS = 12

WEIGHT = 0.9
TRANSITIONS = [ 
    "Receiving Request", 
    "First Assessment", 
    "Fraud Check",  
    "Invisible 2", 
    "Invisible 1",
    "Accept", 
    "Decline", 
    "Create Offer", 
    "Contact Customer", 
    "Offer Refused", 
    "Draw Contract", 
    "Send Contract", 
]

# Load in dataset
dataset = load_dataset(filepath="data/traces.txt", transitions=TRANSITIONS)


##########################################################################################
# Exercise 1
##########################################################################################
outputs_1 = run_petrinas_ga(
    dataset=dataset,
    n_places=N_PLACES,
    n_transitions=N_TRANSITIONS,
    weight=WEIGHT,
    crossover_ops=[
        {"function": tools.cxTwoPoint},
    ],
    mutation_ops=[
        {"function": tools.mutFlipBit, "indpb": 0.05},
    ],
    selection_ops=[
        {"function": tools.selTournament, "tournsize": 5},
    ],
    crossover_probs=[0.5],
    mutation_probs=[0.2],
    n_individuals=100,
    n_generations=100,
    n_iterations=3,
)

# TODO: plotting
