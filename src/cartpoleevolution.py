"""CartpoleEvolution module for project."""
import gym
from sklearn.neural_network import MLPClassifier
import numpy as np
import copy
import random
import pandas as pd


class cartpoleevolution(object):
    """
    Class for simulation of evolutionary agents.

    A population of neural networks solving OpenAI's
    CartPole-v1 problem.
    """

    def __init__(self):
        """Initialise instance of class."""
        self.env = gym.make('CartPole-v1')
        self.population = []
        self.MUTATION_RATE = 0.05

    def initialise_population(self, size):
        """Initialise size number of agents.

        Author: Bjørn Christian Weinbach

        Args:
            size ([int]): [Number of agents in population]

        Returns:
            [population]: [List of agents]
        """
        for _ in range(size):
            self.population.append(self.create_new_network())

    def create_new_network(self):
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
        ).partial_fit(np.array([self.env.observation_space.sample()]),
                      np.array([self.env.action_space.sample()]),
                      classes=np.arange(self.env.action_space.n))

    def breedCrossover(self, index1, index2):
        """Breeds a child from 2 parents using crossover.

        Author: Håvard Godal

        Args:
            index1 (Int): [Index of parent 1]
            index2 ([Int]): [Index of parent 2]

        Returns:
            [children]:
                [List of two children containing coefs_ and intercepts_]
        """
        # Choosing either input -> hidden-layer or hidden-layer -> output
        layer = random.randint(0, 1)
        # layer = 1

        child1 = []
        child2 = []

        for i in range(2):
            if i == 0:
                param1 = self.population[index1].coefs_
                param2 = self.population[index2].coefs_
            else:
                param1 = self.population[index1].intercepts_
                param2 = self.population[index2].intercepts_

            shape = param2[layer].shape

            paramFlat1 = np.ravel(param1[layer])
            paramFlat2 = np.ravel(param2[layer])

            indexes = sorted([int(random.random() * len(paramFlat1)),
                              int(random.random() * len(paramFlat1))])

            # Should consider combining the code chunks beneath into one

            newFlatParam = copy.copy(paramFlat2)
            newFlatParam[indexes[0]:indexes[1]
                         ] = paramFlat1[indexes[0]:indexes[1]]

            newParam = []
            newParam.insert(layer, np.array(newFlatParam).reshape(shape))
            newParam.insert(1 - layer, param2[1 - layer])

            child1.append(newParam)

            ###################################################################

            newFlatParam = copy.copy(paramFlat1)
            newFlatParam[indexes[0]:indexes[1]
                         ] = paramFlat2[indexes[0]:indexes[1]]

            newParam = []
            newParam.insert(layer, np.array(newFlatParam).reshape(shape))
            newParam.insert(1 - layer, param1[1 - layer])

            child2.append(newParam)

        return [child1, child2]

    def mutationFunc_W_B(self, j, method):
        """Mutate agents weights and biases.

        Author:
            Vegard Rongve, Johanna Kinstad, Ove Jørgensen

        Args:
            j (Index): [Index of agent to mutate]
            mutation_rate ([float]): [Probability of mutation]
            method ([ "swap" | "inverse" | "scramble" ]):
                [Type of mutation operation]
        """
        for item in range(2):
            if item == 0:
                node_item = self.population[j].coefs_
            else:
                node_item = copy.copy(self.population[j].intercepts_)

            for el in node_item:
                for swappedRow in el:
                    if (random.random() < self.MUTATION_RATE):
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

    def mutation_rate(self, score):
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
        return(1 - (score/self.env._max_episode_steps))

    def simulate(self, generations=15, agents=50,
                 acceptance_rate=0.95, max_episode_steps=500,
                 mutation="swap"):
        """Run simulation of agents exploring the environment.

        Args:
            generations (int, optional):
                [No of generations]. Defaults to 15.
            population (int, optional):
                [No of agents]. Defaults to 50.
            acceptance_rate (float, optional):
                [Acceptance rate of agents]. Defaults to 0.95.
            max_episode_steps (int, optional):
                [Max iterations in episode]. Defaults to 500.
            mutation (str, optional):
                [Mutation method]. Defaults to "swap".

        Return:
            df (Pandas DataFrame): Dataframe of metrics
        """
        self.initialise_population(agents)
        self.env._max_episode_steps = max_episode_steps
        fit = np.zeros(agents)
        scoreList = np.zeros(100)
        df = pd.DataFrame()

        for i in range(generations):
            for n, agent in enumerate(self.population):
                observation = self.env.reset()
                score = 0
                actions = np.empty(5)
                terminate = False

                j = 0
                while not(terminate):
                    action = int(agent.predict(
                        observation.reshape(1, -1).reshape(1, -1)))
                    if j > 5 and sum(actions) % 5 == 0:
                        action = self.env.action_space.sample()
                    observation, reward, done, _ = self.env.step(action)
                    score += reward
                    j += 1
                    actions[j % 5] = action
                    terminate = done

                fit[n] = score
                scoreList[(agents*i+n) % 100] = score

            score_probability = fit/sum(fit)
            parents = np.argsort(-score_probability)[:2]
            current_best_score = fit[parents[0]]

            # Breed new nn's
            for j in range(0, agents, 2):
                children = self.breedCrossover(parents[0], parents[1])
                for k in range(2):
                    self.population[j+k].coefs_ = children[k][0]
                    self.population[j+k].intercepts_ = children[k][1]

            halved_acceptance_rate = (1 - ((1 - acceptance_rate) / 2))
            comparison = env._max_episode_steps * halved_acceptance_rate
            improvable_network_indices = (fit < comparison).nonzero()[0]
            for j in improvable_network_indices:
                self.mutationFunc_W_B(j, mutation)

            df['Generation ' + str(i)] = fit

            # Terminate if mean score > acceptance_rate
            if (np.mean(scoreList) >=
               self.env._max_episode_steps * acceptance_rate):
                break

        df.index.name = "Agents"
        return df
