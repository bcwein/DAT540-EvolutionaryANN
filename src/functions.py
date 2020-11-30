"""
Python helper functions for training agents.

Functions:
    initialise_population -  Initalises a population of agents
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import random
import copy


def create_new_network(env):
    """Create a new MLPCLassifier.

    Args:
        env: The environment to get training samples from

    Returns:
        MLPClassifier: The new NN partially fitted to
        sample data from the environment
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


# Created a mutation function to mutate both weights and biases for an agent
def mutationFunc_W_B(agent, mutation_rate):
    """Mutation function to mutate both weights and biases for an agent.

    Args:
        agent ([MLPClassifier]): [Neural network of agent]
        mutation_rate ([float]): [probability of mutation]

    Returns:
        [agent]: [Mutated agent]
    """
    for item in range(2):
        if item == 0:
            node_item = agent.coefs_
        else:
            node_item = agent.intercepts_

        for i in range(len(node_item)):
            for swappedRow in range(len(node_item[i])):
                if (random.random() < mutation_rate):
                    rowToSwapWith = int(random.random()*len(node_item[i]))
                    row1 = copy.copy(node_item[i][swappedRow])
                    row2 = copy.copy(node_item[i][rowToSwapWith])
                    node_item[i][swappedRow] = row2
                    node_item[i][rowToSwapWith] = row1

        if item == 0:
            agent.coefs_ = node_item
        else:
            agent.intercepts_ = node_item

    return agent


def breedCrossover(nn1, nn2):
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

    ###################################################################

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
    """Display a simulation of a single given network in a given environment.

    Args:
        network (MLPClassifier): The network to use for simulation
        env (TimeLimit): An OpenAI gym environment in which to
        run the simulation
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
        observation, reward, terminate, _ = env.step(action)
        score += reward
        j += 1
        actions[j % 5] = action
        env.render()
    return score


def average_weight_and_bias(population):
    """Calculate the average weight and bias from a given population.

    Args:
        population: The population from which to calculate

    Returns:
        [coefs]: The average weights of the populaiton
        [intercepts]: The average biases of the population
    """
    coef0 = np.mean(np.array([coef.coefs_[0] for coef in population]), axis=0)
    coef1 = np.mean(np.array([coef.coefs_[1] for coef in population]), axis=0)

    intercept0 = np.mean(
        np.array([intercept.intercepts_[0] for intercept in population]),
        axis=0
    )

    intercept1 = np.mean(
        np.array([intercept.intercepts_[1] for intercept in population]),
        axis=0
    )

    return [coef0, coef1], [intercept0, intercept1]
