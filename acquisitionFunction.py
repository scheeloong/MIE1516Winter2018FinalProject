import math
import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class AcquisitionFunction(object):
    def __init__(self):
        pass

    def greedy(self, expectedValues):
        '''
        Return the greedy item choice
        '''
        return np.argmax(expectedValues)

    def randomlyPickOne(self, expectedValues):
        '''
        randomly returns an integer between [0, uppBound-1]
        '''
        numItems = expectedValues.shape[0]
        validChoices = np.where(expectedValues > 0.0)
        return np.random.choice(validChoices[0])

    def epsilonGreedy(self, expectedValues, epsilon=0.1):
        if (random.random() < epsilon):
            return self.randomlyPickOne(expectedValues)
        else:
            return self.greedy(expectedValues)

    def thompsonSampling(self, sampledItemsForUser, expectedValues):
        '''
        sampledItemsForUser is a dictionary
        '''
        numItems = expectedValues.shape[0]
        validChoices = np.where(expectedValues > 0.0)
        itemChosen = 0
        maxValSoFar = 0.0
        for itemIndex in validChoices[0]:
            sampleVal = np.random.choice(sampledItemsForUser[itemIndex])
            if (sampleVal > maxValSoFar):
                itemChosen = itemIndex
                maxValSoFar = sampleVal
        return itemChosen

    def upperConfidenceBound(self, expectation, stddev):
        '''
        Returns the item with the maximum upper confidence bound
        '''
        return np.argmax(expectation + stddev)
