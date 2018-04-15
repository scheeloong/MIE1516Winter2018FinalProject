import os
import numpy as np
from collections import defaultdict # dictionary of list
import copy
import abc # Abstract Base Class
from abc import ABC, abstractmethod
import math

class RecommenderParser(object):
    def __init__(self, dataDirectory):
        self.dataDirectory = dataDirectory

class MovieLensRating(object):
    '''
    UserItemRating
    Represents a single row in user item matrix
    Used by MovieLensParser class
    '''

    def __init__(self, userId, movieId, rating, timeStamp):
        self.userId= int(userId)
        self.movieId = int(movieId)
        self.rating = float(rating) # 20m dataset has float ratings
        self.timeStamp = int(timeStamp)

    def __eq__(self, other):
        return (isinstance(other, MovieLensRating) and
                self.userId, self.movieId, self.rating, self.timeStamp ==
                other.userId, other.movieId, other.rating, other.timeStamp)

    # Use __lt__ for python3 compatibility.
    def __lt__(self, other):
        '''
        Sort based on timestamp
        '''
        return self.timeStamp < other.timeStamp

    def __hash__(self):
        return hash((self.userId, self.movieId, self.rating, self.timeStamp))

class MovieLensParser(RecommenderParser, ABC):
    # This base class groups common methods in the various movielens datasets
    # Uses template design pattern
    def __init__(self, dataDirectory):
        super().__init__(dataDirectory)

        # MovieId is Id given by the data, which may have skip ids
        # ItemId is the proper id for array indexing which doesn't skip any ids
        # To  map from arbitrary movieId to contiguous unique itemId
        self.movieIdToItemId= dict()
        self.itemIdToMovieId= dict()

        # To map from movieId to movieTitles

        # Abstract methods that must be overriden by child classes
        self.dataFile = self.getDataFileLocation() 
        self.movieFile = self.getMovieFileLocation()
        self.arrOfMovieLensRatings = self.getArrayOfMovieLensRatings() # A single long array
        self.ratingMatrix = self.parseRatingMatrix()
        self.movieIdToName = self.parseMovieIdToTitle()

    @abstractmethod
    def getDataFileLocation(self):
        '''
        Returns location of data file, containing ratings information
        '''
        pass

    @abstractmethod
    def getMovieFileLocation(self):
        '''
        Returns location of movie file, containing movie specific contents
        For instance, getting from movieId to title
        '''
        pass

    @abstractmethod
    def getArrayOfMovieLensRatings(self):
        '''
        Gets array of movie lens rating
        '''
        pass

    @abstractmethod
    def parseRatingMatrix(self):
        '''
        Parse the rating matrix to either 2D matrix or sparse matrix
        '''
        pass

    @abstractmethod
    def parseMovieIdToTitle(self):
        '''
        Parse the movie id's to their corresponding titles
        '''
        pass

    # Public methods
    def parseDictionaryOfMoviesToRatingAndTime(self):
        '''
        Return all movieId to its list of ratings ordered by time
        '''
        movieToListOfMovieLensRating = defaultdict(list)
        for currRating in self.arrOfMovieLensRatings:
            currRating = copy.deepcopy(currRating) # Don't touch the original self.arrOfMovieLensRatings
            currRating.movieId = self.movieIdToItemId[currRating.movieId]
            movieToListOfMovieLensRating[currRating.movieId].append(currRating)
        for key in movieToListOfMovieLensRating:
            movieToListOfMovieLensRating[key].sort()
        return movieToListOfMovieLensRating

    def parseDictionaryOfUsersToRatingAndTime(self):
        ''' Return all userId to its list of ratings ordered by time'''
        userToListOfMovieLensRating = defaultdict(list)
        for currRating in self.arrOfMovieLensRatings:
            currRating = copy.deepcopy(currRating) # Don't touch the original self.arrOfMovieLensRatings
            currRating.movieId = self.movieIdToItemId[currRating.movieId]
            userToListOfMovieLensRating[currRating.userId-1].append(currRating)
        for key in userToListOfMovieLensRating:
            userToListOfMovieLensRating[key].sort()
        return userToListOfMovieLensRating

    def getTrainTestUserRandomly(self, trainTestSplit=0.8):
        '''
        Similar to getTrainTestUserTime, however, 
        splits the movies for each user randomly instead of by time
        the [:trainTestSplit] percentage of time ratings is train, 
        the [trainTestSplit:] percentage of time ratings is test
        '''
        dUser = self.parseDictionaryOfUsersToRatingAndTime()
        dUserTrain = defaultdict(list)
        dUserTest = defaultdict(list)
        for userKey in dUser:
            numRatingForCurrUser = len(dUser[userKey])
            numTrainForCurrUser = int(round(trainTestSplit*numRatingForCurrUser))
            # Randomly shuffle it, no longer based on time
            np.random.shuffle(dUser[userKey])
            dUserTrain[userKey] = dUser[userKey][:numTrainForCurrUser]
            dUserTest[userKey] = dUser[userKey][numTrainForCurrUser:]
        return dUserTrain, dUserTest

    def partitionTestUserTime(self, dUserTest, numTestPartition):
        '''
        Splits the test set into the given number of partitions. 
        The number of partitions must be less than the number of test instances for all users
        '''
        dUserTestPartition = dict()
        for userKey in dUserTest:
            dUserTestPartition[userKey] = defaultdict(list) # to store result
            numTestRating = len(dUserTest[userKey])
            # Use floor as dont want to exceed number of test per partition
            numTestPerPartition = math.floor(float(numTestRating)/float(numTestPartition))
            for i in range(numTestPartition-1):
                dUserTestPartition[userKey][i] = dUserTest[userKey][numTestPerPartition*i: numTestPerPartition*(i+1)]
            # Store everything else in last partition
            dUserTestPartition[userKey][numTestPartition-1] = dUserTest[userKey][numTestPerPartition*(numTestPartition-1):]
        return dUserTestPartition

    def getTrainTestUserTime(self, trainTestSplit=0.8):
        '''
        Split train and test based on:
        user rating time (earlier rating time is train)
        Basically, for every user, 
        the [:trainTestSplit] percentage of time ratings is train, 
        the [trainTestSplit:] percentage of time ratings is test
        '''
        dUser = self.parseDictionaryOfUsersToRatingAndTime()
        dUserTrain = defaultdict(list)
        dUserTest = defaultdict(list)
        for userKey in dUser:
            numRatingForCurrUser = len(dUser[userKey])
            numTrainForCurrUser = int(round(trainTestSplit*numRatingForCurrUser))
            dUserTrain[userKey] = dUser[userKey][:numTrainForCurrUser]
            dUserTest[userKey] = dUser[userKey][numTrainForCurrUser:]
        return dUserTrain, dUserTest

    # TODO: Split into train and test based on: 
    # i) Time
    # ii) Cold User (user in train will NOT be in test)
    # iii) Cold Item (item train will NOT be in test)
    def getTrainTestItemTime(self, trainTestSplit=0.8):
        '''
        Split train and test based on:
        item rating time (earlier rating time is train)
        '''
        raise NotImplementedError("NOT YET IMPLEMENT")

    def getTrainTestColdUser(self, trainTestSplit=0.8):
        raise NotImplementedError("NOT YET IMPLEMENT")

    def getTrainTestColdItem(self, trainTestSplit=0.8):
        raise NotImplementedError("NOT YET IMPLEMENT")

    def getRatingMatrixCopy(self):
        '''
        Get a copy of the 2D rating matrix (users x movies)
        '''
        return self.ratingMatrix.copy()

    def getMovieTitle(self, movieId):
        '''
        Get the movie title of a given movieID
        '''
        return self.movieIdToName[int(movieId)]

    def convertMovieIdToItemId(self, movieId):
        '''
        Given a movieID (unordered, original), return the ordered itemId
        '''
        return int(self.movieIdToItemId[movieId])

class MovieLensParser100k(MovieLensParser):
    '''
    To parse MovieLens 100k version
    '''

    def __init__(self, dataDirectory):
        super().__init__(dataDirectory)
        self.genreFile = os.path.join(self.dataDirectory, 'u.genre')
        self.occupationFile = os.path.join(self.dataDirectory, 'u.occupation')
        self.userFile = os.path.join(self.dataDirectory, 'u.user')

    def getDataFileLocation(self):
        return os.path.join(self.dataDirectory, 'u.data')

    def getMovieFileLocation(self):
        return os.path.join(self.dataDirectory, 'u.item')

    def getArrayOfMovieLensRatings(self):
        # Sort all the ratings by timestamps
        arr = list()
        with open(self.dataFile, mode='r', encoding='iso-8859-1') as dataFile:
            for currLine in dataFile:
                currLine = currLine.strip()
                if currLine:
                    singleRating = MovieLensRating(*tuple(currLine.split()))
                    arr.append(singleRating)
        # Sorted from earlier timestamp to later timestamp
        arr.sort()
        arr = np.array(arr)
        return arr

    def parseRatingMatrix(self):
        ratingMatrix = np.zeros((943, 1682)) # 943 users from 1 to 943, 1682 items based on dataset
        # Start with 0
        uniqueItemId = 0
        for currRating in self.arrOfMovieLensRatings:
            if currRating.movieId not in self.movieIdToItemId:
                self.movieIdToItemId[currRating.movieId] = uniqueItemId
                self.itemIdToMovieId[uniqueItemId] = currRating.movieId
                uniqueItemId += 1
            # Assign currRating
            ratingMatrix[currRating.userId-1][self.movieIdToItemId[currRating.movieId]] = currRating.rating
        print("Number of unique items: ", uniqueItemId)
        return ratingMatrix

    def parseMovieIdToTitle(self):
        movieIdToName = dict()
        with open(self.movieFile, mode='r', encoding='iso-8859-1') as movieFile:
            for currLine in movieFile:
                currLine = currLine.strip()
                if currLine:
                    #  ID, Title, Date, ImdbLink
                    line = currLine.split("|")
                    movieIdToName[int(line[0].strip())] = line[1].strip() 
        return movieIdToName

class MovieLensParser1m(MovieLensParser):
    def __init__(self, dataDirectory):
        super().__init__(dataDirectory)

    def getDataFileLocation(self):
        return os.path.join(self.dataDirectory, 'ratings.dat')

    def getMovieFileLocation(self):
        return os.path.join(self.dataDirectory, 'movies.dat')

    def getArrayOfMovieLensRatings(self):
        # Sort all the ratings by timestamps
        arr = list()
        with open(self.dataFile, mode='r', encoding='iso-8859-1') as dataFile:
            for currLine in dataFile:
                currLine = currLine.strip()
                if currLine:
                    singleRating = MovieLensRating(*tuple(currLine.split('::')))
                    arr.append(singleRating)
        # Sorted from earlier timestamp to later timestamp
        arr.sort()
        arr = np.array(arr)
        return arr
    
    def parseRatingMatrix(self):
        ''' Convert arbitrary movieId to ordered movieIds '''
        # (numUser, numItem)
        # About 23.9 million
        ratingMatrix = np.zeros((6040, 3952)) # For 1m dataset according to readme
        # Start with 0
        uniqueItemId = 0
        for currRating in self.arrOfMovieLensRatings:
            if currRating.movieId not in self.movieIdToItemId:
                self.movieIdToItemId[currRating.movieId] = uniqueItemId
                self.itemIdToMovieId[uniqueItemId] = currRating.movieId
                uniqueItemId += 1
            # Assign currRating
            ratingMatrix[currRating.userId-1][self.movieIdToItemId[currRating.movieId]] = currRating.rating
        print("Number of unique items: ", uniqueItemId)
        return ratingMatrix

    def parseMovieIdToTitle(self):
        movieIdToName = dict()
        with open(self.movieFile, mode='r', encoding='iso-8859-1') as movieFile:
            for currLine in movieFile:
                currLine = currLine.strip()
                if currLine:
                    #  ID, Title, Genre
                    line = currLine.split("::")
                    movieIdToName[int(line[0].strip())] = line[1].strip() + ', ' + line[2].strip()
        return movieIdToName

class MovieLensParser20m(MovieLensParser):
    def __init__(self, dataDirectory):
        super().__init__(dataDirectory)

    def getDataFileLocation(self):
        return os.path.join(self.dataDirectory, 'ratings.csv')

    def getMovieFileLocation(self):
        return os.path.join(self.dataDirectory, 'movies.csv')

    def getArrayOfMovieLensRatings(self):
        # Sort all the ratings by timestamps
        arr = list()
        with open(self.dataFile, mode='r', encoding='iso-8859-1') as dataFile:
            firstLine = True
            for currLine in dataFile:
                if firstLine:
                    firstLine = False
                    continue
                currLine = currLine.strip()
                if currLine:
                    singleRating = MovieLensRating(*tuple(currLine.split(",")))
                    arr.append(singleRating)
        # Sorted from earlier timestamp to later timestamp
        arr.sort()
        arr = np.array(arr)
        return arr

    def parseRatingMatrix(self):
        ratingMatrix = np.zeros((5,5))
        # TODO: Implement this for sparse matrix as original is too large
        '''
        # (numUser, numItem)
        ratingMatrix = np.zeros((138493, 27278))
        # Convert arbitrary movieId to ordered movieIds
        # Start with 0
        uniqueItemId = 0
        for currRating in arr:
            if currRating.movieId in self.movieIdToItemId:
                ratingMatrix[currRating.userId-1][self.movieIdToItemId[currRating.movieId]] = currRating.rating
            else:
                self.movieIdToItemId[currRating.movieId] = uniqueItemId
                uniqueItemId += 1
        '''
        # Start with 0
        uniqueItemId = 0
        for currRating in self.arrOfMovieLensRatings:
            if currRating.movieId not in self.movieIdToItemId:
                self.movieIdToItemId[currRating.movieId] = uniqueItemId
                self.itemIdToMovieId[uniqueItemId] = currRating.movieId
                uniqueItemId += 1
        return ratingMatrix

    def parseMovieIdToTitle(self):
        movieIdToName = dict()
        with open(self.movieFile) as movieFile:
            firstLine = True
            for currLine in movieFile:
                if firstLine:
                    firstLine = False
                    continue
                currLine = currLine.strip()
                if currLine:
                    # movieID, title, genre
                    line = currLine.split(",")
                    movieIdToName[int(line[0].strip())] = line[1].strip() + ", " + line[2].strip()
        return movieIdToName

if __name__ == '__main__':
    trainTestSplit = 0.8
    # Data directories
    installDir = "/root/Github/RecommendationSystems/sclrecommender/data/"
    ## Movielens
    dataDirectory100k = installDir + "movielens/ml-100k"
    dataDirectory1m = installDir + "movielens/ml-1m"
    dataDirectory20m = installDir + "movielens/ml-20m"
    mlp100k = MovieLensParser100k(dataDirectory100k)
    mlp1m = MovieLensParser1m(dataDirectory1m)
    mlp20m = MovieLensParser20m(dataDirectory20m)
    # Data directories
    userDictTrain, userDictTest = mlp100k.getTrainTestUserRandomly(trainTestSplit)
    userDictTrain, userDictTest = mlp100k.getTrainTestUserTime(trainTestSplit)
    userDictTrain, userDictTest = mlp1m.getTrainTestUserRandomly(trainTestSplit)
    userDictTrain, userDictTest = mlp1m.getTrainTestUserTime(trainTestSplit)
    #userDictTrain, userDictTest = mlp20m.getTrainTestUserRandomly(trainTestSplit)
    #userDictTrain, userDictTest = mlp20m.getTrainTestUserTime(trainTestSplit)
