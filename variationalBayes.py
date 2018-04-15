from evaluator import predictRMSE
import os
import numpy as np

# Train U and V using Yee Whye Teh's paper
def trainVariationalBayes(trainDataDict, testDataDict, numItem, numEpoch=10, numHiddenDim = 10):
    '''
    trainDataDict = The initial timestamps for all users
    
    testDataDict = The remaining timestamps for all users
    
    numEpoch = number of epoch to train data 
    
    numHiddenDim = number of latent dimension for both user factors and item factors
    
    Symbols translation: (Refer to Greek letters for link below)
        https://artofproblemsolving.com/wiki/index.php/LaTeX:Symbols
        
    Definitions and Notations from paper: 
        Rating Prediction related
            tausq = variance for observation noise for mean u_{i}v_{j}^T
            
        User related
            U = user latents => used for Q(U) => Variational approximation to maximize P(M)
            - also, Q(U) is used as P(U)
                uBar = mean for users
                Phi = covariance matrix for users
                    sigmasq = entries for variance for users for each different latent dimension = userVariances
                
        Item Related
            V = item latents => used for Q(V) => Variational approximation for P(M)
            - also, Q(V) is used as P(V)
                vBar = mean for items
                    t = observationRating*userMean/varianceForObsvNoise 
                        => summation for terms before calculation of vBar
                      = weightedScaleObservedRatings
                Psy = covariances for items
                    S = Psy inverse = Precision for items (summation terms before inverse for Psy)
                    rho = p = entries for variance for items for each different latent dimension
        Observation = Rating related
            M = observations
    '''
    # Note: DO NOT USE += WITH NUMPY, IT RESULTS IN WEIRD BEHAVIOR THAT YOU AREN'T FAMILIAR WITH YET
    #  Just retype the variable name and use + instead with assignment
    # From:
    #   varA += varB
    # To:
    #   varA = varA + varB
    #-----------------------------------------------------------
    # Hyperparameters Initialization
    #-----------------------------------------------------------
    numUser = len(trainDataDict) 
    #-----------------------------------------------------------
    # Parameters Initialization
    #-----------------------------------------------------------
    varianceObsvNoise = 1 # variance for observation noise
    # Paper 4.2: To reduce redundancy between U and V
    # Helper variables to prevent storing large phi (numUser, hdidenDim, hiddenDim) since number of users is large
    # The original paper worked with Netflix dataset (400k users), which is infeasible
    # You are working with movielens1m, with only 8k users, you can store the users like items as usual
    # Initial condition for variances
    # Initialize to 1  
    userVariances = np.identity(numHiddenDim) 
    # Initialize itemVariances to 1/numHiddenDim 
    # Item variances is a diagonal matrix, although math requires it to be k*k instead of just k
    # TODO: can shrink this to (k, 1) and convert to (k, k) when necessary
    itemVariances = np.zeros((numHiddenDim, numHiddenDim)) + (np.identity(numHiddenDim) * (1.0/numHiddenDim))
    # Initialize itemCovariances as needed for userCovariances calculation
    itemCovariances = np.zeros((numItem, numHiddenDim, numHiddenDim)) + itemVariances
    userCovariances = np.zeros((numUser, numHiddenDim, numHiddenDim)) + userVariances

    # Parameters for user and item
    userLatents = np.random.randn(numUser, numHiddenDim)
    itemLatents = np.random.randn(numItem, numHiddenDim)
    #-----------------------------------------------------------
    # Iterative training
    #-----------------------------------------------------------
    trainRMSE = [0] * (numEpoch+1)
    testRMSE = [0] * (numEpoch+1)
    for trainEpoch in range(numEpoch): 
        trainRmse = predictRMSE(trainDataDict, userLatents, itemLatents)
        testRmse = predictRMSE(testDataDict, userLatents, itemLatents)
        trainRMSE[trainEpoch] = trainRmse
        testRMSE[trainEpoch] = testRmse
        # FIXME: TODO: TEMP TO PREVENT SINGULAR, remove below
        # Note: Original paper did not even train the item variances but instead set it to this constant
        # due to redundancy.
        # You are trying to train the item variances according to the paper but it results in the error:
        # Singular Matrix when you try to train it. 
        # Maybe some values are getting too small? 
        itemVariances = np.zeros((numHiddenDim, numHiddenDim)) + (np.identity(numHiddenDim) * (1.0/numHiddenDim))
        itemVariances = np.zeros((numHiddenDim, numHiddenDim)) + (np.identity(numHiddenDim) * (numHiddenDim))
        # Placeholder for next iteration's variance
        tempUserVariances = np.zeros(userVariances.shape) 
        tempVarianceObsvNoise = 0.0
        tempItemVariances = np.zeros(itemVariances.shape)
        #-----------------------------------------------------------
        # Paper 4.1.1: Initialize Helper variables for computing covarianceForItem for all item 
        #-----------------------------------------------------------
        # Compute final values
        itemPrecision = np.zeros((numItem, numHiddenDim, numHiddenDim))
        scaledWeightedObservedRatingsByUserLatent = np.zeros((numItem, numHiddenDim))  
        #-----------------------------------------------------------
        # Paper 4.1.2 Train for every user
        #-----------------------------------------------------------
        for currUser in range(numUser):
            #-----------------------------------------------------------
            # Paper 4.1.2 a): Update userCovariances and userLatents
            #-----------------------------------------------------------
            userPrecision = np.zeros(userVariances.shape)
            weightedObsvRatingsByItemLatent = np.zeros((numHiddenDim)) 
            # Add each item that is observed for the current user
            for currRating in trainDataDict[currUser]:
                currItem = currRating.movieId
                userPrecision = userPrecision + itemCovariances[currItem] + np.outer(itemLatents[currItem], itemLatents[currItem])
                weightedObsvRatingsByItemLatent = weightedObsvRatingsByItemLatent + (currRating.rating*itemLatents[currItem])
            userPrecision /= varianceObsvNoise
            
            scaledWeightedObsvRatingsByItemLatent = weightedObsvRatingsByItemLatent/varianceObsvNoise
            # Compute the final values
            userCovariances[currUser] = np.linalg.inv(userVariances + userPrecision)
            userLatents[currUser] = np.dot(userCovariances[currUser], scaledWeightedObsvRatingsByItemLatent)
            #-----------------------------------------------------------
            # Paper 4.1.2 b): Update helper variables for itemCovariances
            #-----------------------------------------------------------
            for currRating in trainDataDict[currUser]:
                currItem = currRating.movieId
                # Compute final values
                itemPrecision[currItem] = itemPrecision[currItem] + userCovariances[currUser] + np.outer(userLatents[currUser], userLatents[currUser])
                scaledWeightedObservedRatingsByUserLatent[currItem] = scaledWeightedObservedRatingsByUserLatent[currItem] + ((currRating.rating*userLatents[currUser])/varianceObsvNoise)
                # Update tempVarianceObsvNoise
                # Every term needs to be a scalar
                tempVarianceObsvNoise = (currRating.rating**2.0) \
                - 2.0*currRating.rating*np.dot(userLatents[currUser], itemLatents[currItem]) \
                + np.trace(np.dot((userCovariances[currUser] + np.outer(userLatents[currUser], userLatents[currUser]))
                        , (itemCovariances[currItem] + np.outer(itemLatents[currItem], itemLatents[currItem]))))
            # Update tempUserVariances
            for currHiddenDim in range(numHiddenDim):
                tempUserVariances[currHiddenDim, currHiddenDim] = tempUserVariances[currHiddenDim, currHiddenDim] + userCovariances[currUser][currHiddenDim][currHiddenDim] + np.square(userLatents[currUser][currHiddenDim])
        #-----------------------------------------------------------
        # Paper 4.1.3: Update itemCovariances and itemLatents for all items
        #-----------------------------------------------------------
        for currItem in range(numItem):
            # Compute final values
            itemCovariances[currItem] = np.linalg.inv(itemVariances + (itemPrecision[currItem]/varianceObsvNoise))
            itemLatents[currItem] = np.dot(itemCovariances[currItem], scaledWeightedObservedRatingsByUserLatent[currItem])
            # TODO: FIXME: Not sure if this line is right, but currently end up with singular matrix
            # Update tempItemVariances
            tempItemVariances[currHiddenDim][currHiddenDim] = tempItemVariances[currHiddenDim][currHiddenDim] + itemCovariances[currItem][currHiddenDim][currHiddenDim] +  np.square(itemLatents[currItem][currHiddenDim])
        #-----------------------------------------------------------
        # Paper 4.2: Update variances
        #-----------------------------------------------------------
        # For this paper, the variances are independent of individual users and items
        # The variances only differ based on the latent dimension that they represent.
        tempVarianceObsvNoise /= float(numHiddenDim - 1.0)
        tempUserVariances /= float(numUser - 1.0)
        tempItemVariances /= float(numItem - 1.0)
        # Compute final values
        varianceObsvNoise = tempVarianceObsvNoise
        userVariances = tempUserVariances
        itemVariances = tempItemVariances
    trainRmse = predictRMSE(trainDataDict, userLatents, itemLatents)
    testRmse = predictRMSE(testDataDict, userLatents, itemLatents)
    trainRMSE[numEpoch] = trainRmse
    testRMSE[numEpoch] = testRmse
    # TODO: Not sure if should return variances or covariances
    return userLatents, itemLatents, userVariances, itemVariances, varianceObsvNoise, userCovariances, itemCovariances, trainRMSE, testRMSE
