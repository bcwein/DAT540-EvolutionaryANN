"""
Python helper functions for training agents.

Functions:
    create_new_network - Create new MLP Classifier.
    initialise_population -  Initalises a population of agents
    mutationFunc_W_B -  mutate both weights and biases for an agent
    breedCrossover - Breeds a child from 2 parents using crossover
    show_simulation - Display a simulation of a single given network
    average_weight_and_bias - Calculate the average weight and bias
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import random
import copy
import matplotlib.pyplot as plt


def create_new_network(env):
    """Create a new MLPCLassifier.

    Author: Bjørn Christian Weinbach, Marius Sørensen

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
    """Initialise size number of agents.

    Author: Bjørn Christian Weinbach

    Args:
        size ([int]): [Number of agents in population]

    Returns:
        [population]: [List of agents]
    """
    population = []
    for _ in range(size):
        population.append(create_new_network(env))
    return population


def mutationFunc_W_B(agent, mutation_rate, method):
    """Mutate agents weights and biases.

    Author:
        Vegard Rongve, Johanna Kinstad, Ove Jørgensen

    Args:
        agent ([MLPClassifier]): [Neural Network of agent]
        mutation_rate ([float]): [Probability of mutation]
        method ([ "swap" | "inverse" | "scramble" ]):
            [Type of mutation operation]

    Returns:
        [agent]: [Mutated agent]
    """
    for item in range(2):
        if item == 0:
            node_item = agent.coefs_
        else:
            node_item = copy.copy(agent.intercepts_)

        for el in node_item:
            for swappedRow in el:
                if (random.random() < mutation_rate):
                    random1 = int(random.random()*len(el))
                    random2 = int(random.random()*len(el))
                    if(random1 > random2):
                        random2, random1 = random1, random2

                    if(method == 'swap'):
                        row1 = copy.copy(swappedRow)
                        row2 = copy.copy(el[random1])
                        swappedRow = row2
                        el[random1] = row1

                    elif(method == 'scramble'):
                        random.shuffle(el[random1:random2])

                    elif(method == 'inverse'):
                        el[random1:random2] = el[random1:random2][::-1]

                    elif(method == 'uniform'):
                        randVal = random.random()
                        el[random1] = randVal

    return agent


def breedCrossover(nn1, nn2):
    """Breeds a child from 2 parents using crossover.

    Author: Håvard Godal

    Args:
        nn1 ([MLPClassifier]): [Neural network parent nr 1]
        nn2 ([MLPClassifier]): [Neural network parent nr 2]

    Returns:
        [children]: [List of two children containing coefs_ and intercepts_]
    """
    # Choosing either input -> hidden-layer or hidden-layer -> output
    layer = random.randint(0, 1)
    # layer = 1

    child1 = []
    child2 = []

    for i in range(2):
        if i == 0:
            param1 = nn1.coefs_
            param2 = nn2.coefs_
        else:
            param1 = nn1.intercepts_
            param2 = nn2.intercepts_

        shape = param2[layer].shape

        paramFlat1 = np.ravel(param1[layer])
        paramFlat2 = np.ravel(param2[layer])

        indexes = sorted([int(random.random() * len(paramFlat1)),
                          int(random.random() * len(paramFlat1))])

        # Should consider combining the code chunks beneath into one

        newFlatParam = copy.copy(paramFlat2)
        newFlatParam[indexes[0]:indexes[1]] = paramFlat1[indexes[0]:indexes[1]]

        newParam = []
        newParam.insert(layer, np.array(newFlatParam).reshape(shape))
        newParam.insert(1 - layer, param2[1 - layer])

        child1.append(newParam)

        #######################################################################

        newFlatParam = copy.copy(paramFlat1)
        newFlatParam[indexes[0]:indexes[1]] = paramFlat2[indexes[0]:indexes[1]]

        newParam = []
        newParam.insert(layer, np.array(newFlatParam).reshape(shape))
        newParam.insert(1 - layer, param1[1 - layer])

        child2.append(newParam)

    return [child1, child2]


def show_simulation(network, env):
    """Display a simulation of a single given network in a given environment.

    Author: Marius Sørensen

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


def average_weight_and_bias(population, env):
    """Calculate the average weight and bias from a given population.

    Author: Ove Jørgensen

    Args:
        population: The population from which to calculate

    Returns:
        [avg_network]: A new network given the average bias and weight
    """
    def find_mean(mat, attr_type, i): return np.mean(
        np.array([getattr(el, attr_type)[i] for el in mat]),
        axis=0
    )
    coef0 = find_mean(population, 'coefs_', 0)
    coef1 = find_mean(population, 'coefs_', 1)
    intercept0 = find_mean(population, 'intercepts_', 0)
    intercept1 = find_mean(population, 'intercepts_', 1)

    avg_network = create_new_network(env)
    avg_network.coefs_ = [coef0, coef1]
    avg_network.intercepts_ = [intercept0, intercept1]

    return avg_network


def partial_fit(best_trained, best_network, env):
    """Partial fit neural netowork to actions of best agent.

    Author: Bjørn Christian Weinbach

    Args:
        best_trained (MLPClassifier): [Neural network to be trained]
        best_network (MLPClassifier): [Neural network that scored high]
        env (OpenAI gym): [Environment]

    Returns:
        [best_trained]: [partially fitted neural netork]
    """
    observation = env.reset()
    score = 0
    actions = np.empty(5)
    trainingx = []
    trainingy = []
    actions
    terminate = False

    while not(terminate):
        j = 0
        action = int(best_network.predict(
            observation.reshape(1, -1).reshape(1, -1)))
        if j > 5 and sum(actions) % 5 == 0:
            action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        trainingx.append(observation)
        trainingy.append(action)
        score += reward
        j += 1
        actions[j % 5] = action
        terminate = done

    best_trained.partial_fit(
        trainingx,
        trainingy
    )

    return best_trained


def nnPerformance(generation, best_score, average_score, acceptanceCriteria):
    """Visualize the performance from each generation.

    Author: Vegard Rongve

    Args:
        generation: generation size
        best_score: list with best scores from each generation
        average_score: list with average scores from each generation

    Returns:Plot
    """
    plt.plot(range(1, generation+1), best_score, label="Max score")
    plt.plot(range(1, generation+1), average_score, label="Average score")
    plt.title('Fitness through the generations')
    plt.axhline(y=acceptanceCriteria, color='r',
                linestyle='--', label="Acceptance criteria")
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()


def mutation_rate(score, goal):
    """Dynamic mutation rate.

    Author: Bjørn Christian Weinbach

    Algorithm:
        High mutation rate in beginning -> explores space
        When good agents are found:
            Low mutation rate so new agents are similar to
            previous ones (given that they have a high score)

    Args:
        score (float): [score of best agent]
        goal (float): [Linear function decreasing as score -> goal]
    """
    return(1 - (score/goal))
