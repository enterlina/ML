#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random, numpy, math, copy, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
from math import sqrt
from math import exp
from matplotlib import pyplot
from sklearn import metrics
import pandas as pd
import numpy as np
import itertools

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

from scipy.spatial.distance import cityblock
import numpy as np, random, operator


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, Coordinate):
        xDis = abs(self.x - Coordinate.x)
        yDis = abs(self.y - Coordinate.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "[" + str(self.x) + " " + str(self.y) + "]"


class Get_Distance:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.Get_Distance = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCoordinate = self.route[i]
                toCoordinate = None
                if i + 1 < len(self.route):
                    toCoordinate = self.route[i + 1]
                else:
                    toCoordinate = self.route[0]
                pathDistance += fromCoordinate.distance(toCoordinate)
            self.distance = pathDistance
        return self.distance

    def routeGet_Distance(self):
        if self.Get_Distance == 0:
            self.Get_Distance = 1 / float(self.routeDistance())
        return self.Get_Distance


def createRoute(vectors):
    route = random.sample(vectors, len(vectors))
    return route


def initialPopulation(popSize, vectors):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(vectors))
    return population


def rankRoutes(population):
    Get_DistanceResults = {}
    for i in range(0, len(population)):
        Get_DistanceResults[i] = Get_Distance(population[i]).routeGet_Distance()
    return sorted(Get_DistanceResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Get_Distance"])
    df['cum_sum'] = df.Get_Distance.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Get_Distance.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            Coordinate1 = individual[swapped]
            Coordinate2 = individual[swapWith]

            individual[swapped] = Coordinate2
            individual[swapWith] = Coordinate1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Min path= " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


vectors = []

df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW9/tsp.csv')
y = df['label']
y = np.array(y)
X = df.drop(['label'], axis=1)
X = np.array(X)

for i in range(0, len(X)):
    vectors.append(Coordinate(x=X[i][0], y=X[i][1]))

a = vectors[0]

bestRoute = geneticAlgorithm(population=vectors, popSize=80, eliteSize=10, mutationRate=0.01, generations=1000)
vectors = np.asarray(vectors)
bestRoute = np.asarray(bestRoute)
way = []

for i in range(len(vectors)):
    temp = [0, 0]
    t_x = bestRoute[i].x
    t_y = bestRoute[i].y

    temp[0] = t_x
    temp[1] = t_y
    way.append(temp)

x, y = [], []
x.append(way[0][0])
y.append(way[0][1])
for i in range(1, len(way)):
    x.append(way[i - 1][0])
    y.append(way[i][1])
    x.append(way[i][0])
    y.append(way[i][1])

x_t = []
y_t = []
for i in range(len(way)):
    x_t.append(way[i][0])
    y_t.append(way[i][1])

plt.plot(x_t, y_t, 'xb', x, y, 'b')
plt.title('Genetic Algorithm')
plt.show()


# In[ ]:




