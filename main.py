##########################################################################################
# Imports
##########################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

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
        {"function": tools.mutFlipBit, "indpb": 0.1},
    ],
    selection_ops=[
        {"function": tools.selTournament, "tournsize": 5},
    ],
    crossover_probs=[0.5],
    mutation_probs=[0.5],
    n_individuals=500,
    n_generations=50,
    n_iterations=1,
)

fig, ax = plt.subplots(dpi=250, figsize=(8, 5))
avg_fitness = outputs_1[0]["logbook"].select("avg")
best_fitness = outputs_1[0]["logbook"].select("max")
ax.plot(avg_fitness, label="Average fitness")
ax.plot(best_fitness, label="Best fitness")
ax.set_title("Fitness at each generation (1 run)")
ax.set_xlabel("Generation")
ax.legend()
fig.savefig("plots/exercise-1-fitness.png")
