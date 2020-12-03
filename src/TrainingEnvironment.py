"""Python script for training generations of neural networks."""

import gym
import numpy as np
import functions
import copy
import wandb
#wandb.init(project="GymVisualizationFinal")
methods =['swap', 'scramble', 'inverse', 'uniform', 'gaussian']
for method in methods:
    for run in range(10):
        wandb.init(project="GymVisualizationFinal", reinit= True, group=method)
        population_size = 50
        generations = 15
        avgAgents = []
        global_best_score = 0
        scoreList = np.zeros(100)

        env = gym.make('CartPole-v1')
        env._max_episode_steps = 500
        acceptance_rate = 0.95

        listOfAverageScores = []
        listOfBestScores = []

        # Trained agent
        best_trained = functions.create_new_network(env)

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
                j = 0
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
                fit[n] = score

                scoreList[(population_size*i+n) % 100] = score
            
            current_best_index = np.argmax(fit)
            current_best_score = fit[current_best_index]

            wandb.log({'avg': np.mean(scoreList), 'best': current_best_score})
            wandb.save("mymodel.h5")
            if np.mean(scoreList) >= env._max_episode_steps*acceptance_rate:
                print(" " * (population_size + 2), end="\r")
                print(f"\nSuccess in generation {i+1}!")
                print(f"Current average score: {np.mean(scoreList)}")
                np.set_printoptions(suppress=True)
                # print(scoreList)
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
                best_trained = functions.partial_fit(best_trained,
                                                    best_network,
                                                    env)

            # Breed new nn's
            for j in range(0, int(population_size), 2):
                children = functions.breedCrossover(parent1, parent2)
                for k in range(2):
                    population[j+k].coefs_ = children[k][0]
                    population[j+k].intercepts_ = children[k][1]

            for j in range(population_size):
                population[j] = functions.mutationFunc_W_B( population[j],
                                                            functions.mutation_rate(
                                                                current_best_score,
                                                                env._max_episode_steps),
                                                            method
                )

            print(" " * (population_size + 2), end="\r")
            print(
                f'Gen {i+1}: Average: {np.average(fit)} | Best: {current_best_score}'
            )
            
            # Network based on average weight and bias over all levels
            # avgAgent = functions.average_weight_and_bias(avgAgents, env)

            listOfAverageScores.append(np.average(fit))
            listOfBestScores.append(current_best_score)

        # Render of best, average and trained network
        # functions.show_simulation(best_network, env)
        # functions.show_simulation(avgAgent, env)
        # functions.show_simulation(best_trained, env)

        #functions.nnPerformance(len(listOfBestScores),
                                #listOfBestScores, listOfAverageScores, env._max_episode_steps*acceptance_rate)

        env.close()
