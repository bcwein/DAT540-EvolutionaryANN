"""
Python helper functions for training agents.

Functions:
    initialise_population -  Initalises a population of agents
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import random
import copy


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


def swapMutation(coefL_crossover, mutationRate):

    for swapped in range(len(coefL_crossover)):

        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(coefL_crossover))

            coef1 = coefL_crossover[swapped]
            coef2 = coefL_crossover[swapWith]

            coefL_crossover[swapped] = coef2
            coefL_crossover[swapWith] = coef1
    return coefL_crossover

def mutationFunc_W_B(agent,mutation_rate): #Created a mutation function to mutate both weights and biases for an agent
    for item in range(2):
        if item == 0:
            node_item = agent.coefs_
        else:
            node_item = agent.intercepts_
           
        for i in range(len(node_item)):
            for swappedRow in range(len(node_item[i])):
                if (random.random()<mutation_rate):
                    rowToSwapWith = int(random.random()*len(node_item[i]))
                    row1 = copy.copy(node_item[i][swappedRow])
                    #print(row1)
                    row2 = copy.copy(node_item[i][rowToSwapWith])
                    #print(row2)
                    node_item[i][swappedRow] = row2
                    node_item[i][rowToSwapWith] = row1
                   
        if item == 0:            
            agent.coefs_ = node_item
        else:
            agent.intercepts_ = node_item

    return agent



def breedCrossover(nn2, nn1):
    """[Breeds a child from 2 parents using crossover].

    Args:
        nn1 ([MLPClassifier]): [Neural network parent nr 1]
        nn2 ([MLPClassifier]): [Neural network parent nr 2]

    Returns:
        [newcoef, newinter]: [List of coefs_ and intercepts_]
    """
    layer = random.randint(0, 1)
    shape = nn2.coefs_[layer].shape

    coefFlat1 = np.ravel(nn1.coefs_[layer])
    coefFlat2 = np.ravel(nn2.coefs_[layer])

    indexes = sorted([int(random.random() * len(coefFlat1)),
                      int(random.random() * len(coefFlat1))])

    coefFlat2[indexes[0]:indexes[1]] = coefFlat1[indexes[0]:indexes[1]]

    newCoefs = []
    newCoefs.insert(layer, np.array(coefFlat2).reshape(shape))
    newCoefs.insert(1 - layer, nn2.coefs_[1 - layer])

    ###########################################################################

    shape = nn2.intercepts_[layer].shape

    interFlat1 = np.ravel(nn1.intercepts_[layer])
    interFlat2 = np.ravel(nn2.intercepts_[layer])

    indexes = sorted([int(random.random() * len(coefFlat1)),
                      int(random.random() * len(coefFlat1))])

    interFlat2[indexes[0]:indexes[1]] = interFlat1[indexes[0]:indexes[1]]

    newInters = []
    newInters.insert(layer, np.array(interFlat2).reshape(shape))
    newInters.insert(1 - layer, nn2.intercepts_[1 - layer])

    return newCoefs, newInters
