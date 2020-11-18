"""Python script for training generations of neural networks."""

import gym
import numpy as np
import functions

population_size = 50
generations = 1
mutation_rate = 20

env = gym.make('CartPole-v0')

population = functions.initialise_population(population_size, env)
fit = np.zeros(population_size)
for i in range(generations):
    for n, agent in enumerate(population):
        observation = env.reset()
        score = 0
        for j in range(1000):
            observation, reward, done, info = env.step(
                int(agent.predict(observation.reshape(1, -1).reshape(1, -1)))
            )
            score += reward
            if done:
                break

        fit[n] = score

print(fit)
env.close()
