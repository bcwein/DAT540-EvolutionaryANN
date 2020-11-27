"""Python script for training generations of neural networks."""

import gym
import numpy as np
import functions
import copy

population_size = 50
generations = 4  # 15
mutation_rate = 0.001

env = gym.make('CartPole-v1')
env._max_episode_steps = np.inf

population = functions.initialise_population(population_size, env)
fit = np.zeros(population_size)
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
            # env.render()
            terminate = done

        fit[n] = score

    score_probability = fit/sum(fit)
    parents_index = np.argsort(-score_probability)[:2]

    parent1 = copy.copy(population[parents_index[0]])
    parent2 = copy.copy(population[parents_index[1]])

    # Breed new nn's
    for j in range(population_size):
        newCoef, newInter = functions.breedArch(parent1, parent2)
        population[j].coefs_ = newCoef
        population[j].intercepts_ = newInter

print(fit)
env.close()
