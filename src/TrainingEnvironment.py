"""
Python script for training generations of neural networks.

This script has been worked on by all the authors
and ownership is among all the authors.

Otherwise, using vscode's Gitlens extension provides a way to see
who commited each line.

Authors:
    Bjørn Christian Weinbach:
    Marius Sørensen:
    Ove Jørgensen:
    Håvard Godal:
    Johanna Kinstad:
    Vegard Rongve:
"""

import gym
import numpy as np
import functions
import copy

# Support Metrics
avgAgents = []
global_best_score = 0
scoreList = np.zeros(100)
listOfAverageScores = []
listOfBestScores = []
goalReached = False

# Environment
env = gym.make('CartPole-v1')

# Hyperparameters
env._max_episode_steps = 500
acceptance_rate = 0.95
population_size = 50
generations = 15
mutation_type = "swap"

current_best_score = 0

# Population
population = functions.initialise_population(population_size, env)
fit = np.zeros(population_size)
max_score = 0

# Iterate over generations
for i in range(generations):
    # Iterate over agents.
    for n, agent in enumerate(population):
        # Loading bar
        print(
            "[" + "="*(n + 1) + " "*(population_size - n - 1) + "]", end="\r"
        )
        # Simulate the current agent
        score = functions.simulate_agent(agent, env)
        # Store fitness.
        fit[n] = score
        scoreList[(population_size*i+n) % 100] = score

    best_agents_indexes = np.argsort(-fit)[:2]

    # Termination when acceptance rate achieved.
    if np.mean(scoreList) >= env._max_episode_steps*acceptance_rate:
        print(" " * (population_size + 2), end="\r")
        print(f"\nSuccess in generation {i+1}!")
        print(f"Current average score: {np.mean(scoreList)}")
        np.set_printoptions(suppress=True)
        # Render best agent.
        # functions.simulate_agent(population[best_agents_indexes[0]], env, True)
        break

    parent1 = copy.copy(population[best_agents_indexes[0]])
    parent2 = copy.copy(population[best_agents_indexes[1]])

    current_best_index = np.argmax(fit)
    current_best_score = fit[current_best_index]

    # Store current global minimum
    if(current_best_score >= max_score):
        max_score = current_best_score
        best_network = copy.copy(population[current_best_index])

    # Breed new agents
    for j in range(population_size):
        newCoefs, newIntercepts = functions.de_crossover(parent1, parent2)
        population[j].coefs_ = newCoefs
        population[j].intercepts_ = newIntercepts

    # Mutate agents that does not perform well.
    halved_acceptance_rate = (1 - ((1 - acceptance_rate) / 2))
    comparison = env._max_episode_steps * halved_acceptance_rate
    improvable_network_indices = (fit < comparison).nonzero()[0]
    for j in improvable_network_indices:
        population[j] = functions.mutationFunc_W_B(
            population[j],
            0.1,
            mutation_type
        )

    print(" " * (population_size + 2), end="\r")
    print(
        f'Gen {i+1}: Average: {np.average(fit)} | Best: {current_best_score}'
    )

    avgAgent = functions.average_weight_and_bias(population, env)
    avgAgents.append(avgAgent)

    listOfAverageScores.append(np.average(fit))
    listOfBestScores.append(current_best_score)

functions.nnPerformance(len(listOfBestScores),
                        listOfBestScores,
                        listOfAverageScores,
                        env._max_episode_steps*acceptance_rate)

env.close()
