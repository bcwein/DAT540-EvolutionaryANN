"""Python script for training generations of neural networks."""

import gym
import numpy as np
import functions
import copy

population_size = 50
generations = 10
mutation_rate = 0.05  # 0.001
max_training = 50000
avgAgents = []
global_best_score = 0

env = gym.make('CartPole-v1')
env._max_episode_steps = np.inf

population = functions.initialise_population(population_size, env)
fit = np.zeros(population_size)
max_score = 0

for i in range(generations):
    for n, agent in enumerate(population):
        observation = env.reset()
        score = 0
        actions = np.empty(5)
        terminate = False
        print(
            "[" + "="*(n + 1) + " "*(population_size - n - 1) + "]", end="\r"
        )
        while not(terminate):
            j = 0
            action = int(agent.predict(
                observation.reshape(1, -1).reshape(1, -1)))
            if j > 5 and sum(actions) % 5 == 0:
                action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            score += reward
            j += 1
            actions[j % 5] = action
            terminate = done
            terminate = True if score > max_training else terminate
        fit[n] = score

    score_probability = fit/sum(fit)
    parents_index = np.argsort(-score_probability)[:2]

    parent1 = copy.copy(population[parents_index[0]])
    parent2 = copy.copy(population[parents_index[1]])

    avgCoefs, avgIntercepts = functions.average_weight_and_bias(population)
    avgAgent = functions.create_new_network(env)
    avgAgent.coefs_ = avgCoefs
    avgAgent.intercepts_ = avgIntercepts
    avgAgents.append(avgAgent)

    # Breed new nn's
    for j in range(0, int(population_size/2), 2):
        children = functions.breedCrossover(parent1, parent2)
        for k in range(2):
            population[j].coefs_ = children[k][0]
            population[j].intercepts_ = children[k][1]

    for j in range(population_size):
        population[j] = functions.mutationFunc_W_B(population[j],
                                                   mutation_rate, 
                                                   'swap')

    current_best_index = np.argmax(fit)
    current_best_score = fit[current_best_index]

    # Store current global minimum
    if(current_best_score > max_score):
        max_score = current_best_score
        best_network = copy.copy(population[current_best_index])

    print(" " * (population_size + 2), end="\r")
    print(
        f'Gen {i+1}: Average: {np.average(fit)} | Best: {current_best_score}'
    )

# Network based on average weight and bias over all levels
avgCoef, avgIntercept = functions.average_weight_and_bias(avgAgents)
avgAgent = functions.create_new_network(env)
avgAgent.coefs_ = avgCoef
avgAgent.intercepts_ = avgIntercept

# Render of best and average network
# functions.show_simulation(best_network, env)
# functions.show_simulation(avgAgent, env)

env.close()
