"""Python script for training generations of neural networks."""

import gym
import numpy as np
import functions
import copy

population_size = 50
generations = 5 # 15
mutation_rate = 0.001

env = gym.make('CartPole-v1')
env._max_episode_steps = np.inf

population = functions.initialise_population(population_size, env)
fit = np.zeros(population_size)
max_score = 0
best_network = None
for i in range(generations):
    for n, agent in enumerate(population):
        observation = env.reset()
        score = 0
        actions = np.empty(5)
        terminate = False
        while not(terminate):
            j = 0
            action = int(agent.predict(
                observation.reshape(1, -1).reshape(1, -1)))
            if j > 5 and sum(actions) % 5 == 0:
                action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score += reward
            j += 1
            actions[j % 5] = action
            terminate = done

        fit[n] = score

    score_probability = fit/sum(fit)
    parents_index = np.argsort(-score_probability)[:2]

    parent1 = copy.copy(population[parents_index[0]])
    parent2 = copy.copy(population[parents_index[1]])

    # Breed new nn's
    for j in range(population_size):
        newCoef, newInter = functions.breedCrossover(parent1, parent2)
        population[j].coefs_ = newCoef
        population[j].intercepts_ = newInter


    current_best_index = np.argmax(fit)
    current_best_score = fit[current_best_index]

    if(current_best_score > max_score):
        max_score = current_best_score
        best_network = population[current_best_index]

    print(f'Gen {i}: Average: {np.average(fit)} | Best: {current_best_score}')

functions.show_simulation(best_network, env)

# Network based on average weight and bias over all levels
average_network = functions.average_weight_and_bias(population, env)
functions.show_simulation(average_network, env)

env.close()