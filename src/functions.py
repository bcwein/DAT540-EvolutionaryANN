"""
Python helper functions for training agents.

Functions:
    initialise_population -  Initalises a population of agents
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import random


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


def swapMutation(weights):
    """[Mutate weights of parents to avoid local minima].

    Algorithm:
        1. Select two random pair of rows and indexes.
        2. Swap the weights of the two indexes.
        3. (to be implemented) do the same for bias.

    Args:
        weights [numpy array]: [Weights of agent]

    Returns:
        weights [numpy array]: [the new weights for agent]
    """
    layer1 = np.random.randint(0,  len(weights))
    row1 = np.random.randint(0,  len(weights[0]))
    layer2 = np.random.randint(0,  len(weights))
    row2 = np.random.randint(0,  len(weights[0]))
    tmp = weights[[layer1, row1]]
    weights[[layer1, row1]] = weights[[layer2, row2]]
    weights[[layer2, row2]] = tmp

    return weights


def breedArch(nn1, nn2):
    """[Breeds a child from 2 parents using crossover].

    Args:
        nn1 ([MLPClassifier]): [Neural network parent nr 1]
        nn2 ([MLPClassifier]): [Neural network parent nr 2]

    Returns:
        [newcoef, newinter]: [List of coefs_ and intercepts_]
    """
    coef1 = nn1.coefs_
    coef2 = nn2.coefs_
    inter1 = nn1.intercepts_
    inter2 = nn2.intercepts_
    newcoef = []
    newinter = []

    for i in range(min(len(coef1), len(coef2))):
        if random.random() >= 0.5:
            newcoef.append(coef1[i])
        else:
            newcoef.append(coef2[i])

        if random.random() >= 0.5:
            newinter.append(inter1[i])
        else:
            newinter.append(inter2[i])

    return newcoef, newinter
