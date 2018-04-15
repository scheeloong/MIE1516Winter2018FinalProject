import numpy as np
import math

def predictRMSE(testDataDict, userLatents, itemLatents): 
    SE = 0.0
    numRating = 0
    for currUser in testDataDict:
        for currRating in testDataDict[currUser]:
            currItem = currRating.movieId
            SE += np.square(currRating.rating - np.dot(userLatents[currUser], itemLatents[currItem]))
            numRating += 1
    MSE = SE/float(numRating)
    RMSE = math.sqrt(MSE)
    return RMSE
