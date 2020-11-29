"""
Python helper functions for training agents.

Functions:
    initialise_population -  Initalises a population of agents
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import random

def create_new_network(env):
    """ Creates a new MLPCLassifier
    Args:
        env: The environment to get training samples from

    Returns:
        MLPClassifier: The new NN partially fitted to sample data from the environment
    """
    return MLPClassifier(
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

def initialise_population(size, env):
    """[Initialise size number of agents].

    Args:
        size ([int]): [Number of agents in population]

    Returns:
        [population]: [List of agents]
    """
    population = []
    for _ in range(size):
        population.append(create_new_network(env))
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

def show_simulation(network, env):
    """
        Displays a simulation of a single given network in a given environment

        Parameters:
            network (MLPClassifier): The network to use for simulation
            env (TimeLimit): An OpenAI gym environment in which to run the simulation 
    """
    observation = env.reset()
    score = 0
    actions = np.empty(5)
    terminate = False
    while not(terminate):
        j = 0
        action = int(network.predict(
            observation.reshape(1, -1).reshape(1, -1)))
        if j > 5 and sum(actions) % 5 == 0:
            action = env.action_space.sample()
        observation, reward, terminate, info = env.step(action)
        score += reward
        j += 1
        actions[j % 5] = action
        env.render()
    return score

def average_weight_and_bias(population, env):
    """ Creates a new MLPCLassifier
    Args:
        env: The environment to get training samples from

    Returns:
        average_network: A new NN created using the MLPClassifier
    """
    coef0 = np.mean(np.array([coef.coefs_[0] for coef in population]), axis=0)
    coef1 = np.mean(np.array([coef.coefs_[1] for coef in population]), axis=0)
    average_weight = [coef0, coef1]
    
    intercept0 = np.mean(np.array([intercept.intercepts_[0] for intercept in population]), axis=0)
    intercept1 = np.mean(np.array([intercept.intercepts_[1] for intercept in population]), axis=0)
    average_bias = [intercept0, intercept1]

    # Create new network with the averages
    average_network = create_new_network(env)
    average_network.coefs_ = average_weight
    average_network.intercepts_ = average_bias

    return average_network