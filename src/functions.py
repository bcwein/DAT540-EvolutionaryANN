"""
Python helper functions for training agents.

Functions:
    initialise_population -  Initalises a population of agents
"""

from sklearn.neural_network import MLPClassifier
import numpy as np


def initialise_population(size, env):
    """[Initialise size number of agents].

    Args:
        size ([int]): [Number of agents in population]

    Returns:
        [population]: [List of agents]
    """
    population = []
    for _ in range(size):
        population.append(
            MLPClassifier(
                batch_size=1,
                max_iter=1,
                solver='sgd',
                activation='relu',
                learning_rate='invscaling',
                hidden_layer_sizes=4,
                random_state=1
            ).partial_fit(np.array([env.observation_space.sample()]),
                          np.array([env.action_space.sample()]),
                          classes=np.arange(env.action_space.n))
        )
    return population
