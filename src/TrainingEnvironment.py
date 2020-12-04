"""Python script for training generations of neural networks."""

import gym
import numpy as np
import functions
import copy
import wandb

# methods =['swap', 'scramble', 'inverse', 'uniform', 'gaussian']
# for method in methods:
for run in range(50):
    wandb.init(project="GymVisualizationSwap", reinit= True, group='swap2')
    # Support Metrics
    avgAgents = []
    global_best_score = 0
    scoreList = np.zeros(100)
    listOfAverageScores = []
    listOfBestScores = []

    # Environment
    env = gym.make('CartPole-v1')

    # Hyperparameters
    env._max_episode_steps = 500
    acceptance_rate = 0.95
    population_size = 50
    generations = 15
    mutation = "swap"

    # Population
    population = functions.initialise_population(population_size, env)
    fit = np.zeros(population_size)
    max_score = 0

    # Iterate over generations
    for i in range(generations):
        # Iterate over agents.
        for n, agent in enumerate(population):
            observation = env.reset()
            score = 0
            actions = np.empty(5)
            terminate = False
            # Loading bar
            print(
                "[" + "="*(n + 1) + " "*(population_size - n - 1) + "]", end="\r"
            )
            j = 0
            # Agent-Environment Interaction
            while not(terminate):
                action = int(agent.predict(
                    observation.reshape(1, -1).reshape(1, -1)))
                if j > 5 and sum(actions) % 5 == 0:
                    action = env.action_space.sample()
                observation, reward, done, _ = env.step(action)
                score += reward
                j += 1
                actions[j % 5] = action
                terminate = done
            # Store performance.
            fit[n] = score
            scoreList[(population_size*i+n) % 100] = score

        current_best_index = np.argmax(fit)
        current_best_score = fit[current_best_index]
        wandb.log({'avg': np.mean(scoreList), 'best': current_best_score})
        wandb.save("mymodel.h5")
        # Termination when acceptance rate achieved.
        if np.mean(scoreList) >= env._max_episode_steps*acceptance_rate:
            print(" " * (population_size + 2), end="\r")
            print(f"\nSuccess in generation {i+1}!")
            print(f"Current average score: {np.mean(scoreList)}")
            np.set_printoptions(suppress=True)
            # Render best agent.
            #functions.show_simulation(population[parents_index[0]], env)

            listOfAverageScores.append(np.average(fit))
            listOfBestScores.append(current_best_score)
            break

        score_probability = fit/sum(fit)
        parents_index = np.argsort(-score_probability)[:2]

        parent1 = copy.copy(population[parents_index[0]])
        parent2 = copy.copy(population[parents_index[1]])

        avgAgent = functions.average_weight_and_bias(population, env)
        avgAgents.append(avgAgent)

        

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
                0.05,
                mutation
            )

        print(" " * (population_size + 2), end="\r")
        print(
            f'Gen {i+1}: Average: {np.average(fit)} | Best: {current_best_score}'
        )

        listOfAverageScores.append(np.average(fit))
        listOfBestScores.append(current_best_score)

    #functions.nnPerformance(len(listOfBestScores),
                           # listOfBestScores,
                            #listOfAverageScores,
                            #env._max_episode_steps*acceptance_rate)

    env.close()