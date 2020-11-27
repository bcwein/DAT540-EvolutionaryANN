"""Python script for training generations of neural networks."""

import gym
import numpy as np
import functions

population_size = 50
generations = 1  # 15
mutation_rate = 0.001

env = gym.make('CartPole-v1')

population = functions.initialise_population(population_size, env)
fit = np.zeros(population_size)
for i in range(generations):
    for n, agent in enumerate(population):
        observation = env.reset()
        score = 0
        actions = np.empty(5)
        for j in range(500):
            action = int(agent.predict(
                observation.reshape(1, -1).reshape(1, -1)))
            if j >= 5 and sum(actions) % 5 == 0:
                action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score += reward
            actions[j % 5] = action
            # env.render()
            if done:
                break

        fit[n] = score

print(fit)
env.close()
