import math
import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class PosteriorEstimation(object):
    def __init__(self, userLatents, itemLatents, userVariances, itemVariances):
        self.userLatents = userLatents
        self.itemLatents = itemLatents
        self.userVariances = userVariances
        self.itemVariances = itemVariances
        self.numLatentDim = userLatents.shape[1]

    def plotHist(self, samples, plotIndex):
        plt.figure(plotIndex)
        plt.title(str(plotIndex) + ", expectation: " + str(self.expectation(samples)))
        plt.hist(samples)
        plt.show()

    def expectation(self, samples):
        '''
        Calculate expectation for samples
        '''
        return np.mean(samples)

    def stddev(self, samples):
        '''
        Calculate standard deviation for samples
        '''
        return np.std(samples)

class HistogramSamplingPosteriorEstimation(PosteriorEstimation):
    def __init__(self, userLatents, itemLatents, userVariances, itemVariances):
        super().__init__(userLatents, itemLatents, userVariances, itemVariances)

    def sampleForUserAndItem(self, userIndex, itemIndex, numSamples):
        '''
        Sample  the posterior
        '''
        resultSamples = np.zeros(numSamples) 
        for currDim in range(self.numLatentDim):
            samplesUser = np.random.normal(self.userLatents[userIndex][currDim], self.userVariances[currDim][currDim], numSamples) 
            samplesItem = np.random.normal(self.itemLatents[itemIndex][currDim], self.itemVariances[currDim][currDim], numSamples) 
            resultSamples += samplesUser * samplesItem
        return np.array(resultSamples)

class MomentMatchingPosteriorEstimation(PosteriorEstimation):
    def __init__(self, userLatents, itemLatents, userVariances, itemVariances):
        '''
        This does lazy evaluation.
        However, might be faster to just evaluate moments for all user and item pairs initially,
        then access them when needed.
        '''
        super().__init__(userLatents, itemLatents, userVariances, itemVariances)

    def getMoments(self, userIndex, itemIndex):
        '''
        Calculate the moments
        '''
        momentMean = 0.0
        momentSqrMean = 0.0
        for currDim in range(self.numLatentDim):
            # (E[X])
            momentMean +=  self.userLatents[userIndex][currDim] * self.itemLatents[itemIndex][currDim]
            # (E[X^2])
            momentSqrMean += math.pow(self.userLatents[userIndex][currDim] * self.itemLatents[itemIndex][currDim], 2.0)
        # E[X^2] - (E[X])^2
        momentVariance = momentSqrMean - math.pow(momentMean, 2.0)/float(self.numLatentDim)
        return momentMean, momentVariance

    def sampleForUserAndItem(self, userIndex, itemIndex, numSamples):
        '''
        Get the mean and variance of posterior using lazy evaluation
        Then, calculate samples from a Normal Distribution
        '''
        momentMean, momentVariance = self.getMoments(userIndex, itemIndex)
        resultSamples = np.zeros(numSamples) 
        # Moment match from a normal distribution
        resultSamples = np.random.normal(momentMean, math.sqrt(momentVariance), numSamples) 
        return np.array(resultSamples)
