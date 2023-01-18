'''
MATH 122A Final Project Code For Six Methods in Part 2

@ Allen Minch, Marco Qin, and Taku Hagiwara

'''

from keras.datasets import boston_housing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

# load the data set
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# given i points in a numpy array called dataset, and j training cluster centers, for each i, 
# compute the distance to each of the j training cluster centers

# the reason to generalize and use dataset as a parameter instead of just x_test is because for the
# methods (4, 5, and 6) that perform clustering on the test data, the relevant dataset will be just the cluster centers of
# the test data, not all of the test data
def computeDistances(dataset, kmeans, num):
    distances = np.zeros((len(dataset), num))
    for i in range(len(dataset)):
        for j in range(num):
            distances[i][j] = np.linalg.norm(dataset[i] - kmeans.cluster_centers_[j])
    return distances


# given i points in a numpy array called dataset, and j training cluster centers, determine for each i which single training cluster
# center is closest to the ith point in order to decide what model to use to make a prediction at each of the i points

# again, the reason for generalizing here is because dataset will simply be a set of test cluster centers rather
# than all test points in methods 4, 5, and 6
def get_models(dataset, kmeans, num):
    distances = computeDistances(dataset, kmeans, num)
    modelsToUse = np.zeros(len(dataset))
    for i in range(len(dataset)):
        # select the model trained using data with the closest training cluster center to the given point
        modelsToUse[i] = np.argmin(distances[i])
    return modelsToUse


# Option A, Rule 1
# Method 1 - the streamlined version of Part 1 part (iii) of the project
# all weight on closest training cluster center, no test clustering
# KMeans object and the number of clusters, num, are passed as parameters
def total_mse1(kmeans, num):
    # figure out which model to use to make a prediction for each point in x_test
    models = get_models(x_test, kmeans, num)
    total = 0
    for i in range(num):
        # make a Ridge object
        ridge = Ridge()
        # fit it only using training data from the cluster of the training data with label i
        ridge.fit(x_train[kmeans.labels_==i], y_train[kmeans.labels_==i])
        # we don't need to do anything more with the ith model if it turns out that none of the test points are closest to it
        if x_test[models==i].shape[0] != 0:
            # make predictions for y at all test points for which model i is supposed to be used to predict y
            y_pred = ridge.predict(x_test[models==i])
            # get the corresponding true values of y
            y_true = y_test[models == i]
            for j in range(len(y_true)):
                # add the squared errors in all of the values of y predicted using model i
                total += (np.linalg.norm(y_true[j] - y_pred[j]))**2
    # total squared error divided by the number of data points
    return total / len(x_test)

# this function will be useful for Methods 2, 3, 5, and 6
# given a set of distances or squared distances from each of i points to each of 
# j training cluster centers, then for each i, calculate harmonic weights for each cluster center j,
# meaning that for each i, the weight given to cluster center j times the distance (or distance squared) to cluster center j
# is constant (weight is inversely proportional to distance (or distance squared))
def calculateWeights(distances):
    # weights is a matrix (numpy array) whose ij entry is the weight given to cluster center j for point i
    # thus, it has the same shape as distances
    weights = np.zeros(distances.shape)
    # to calculate the values in each row of weights, we only need focus on the distances in the corresponding row of distances
    for i in range(len(distances)):
        # to find the harmonic weight that should be given to cluster center k for point i, one must divide the 
        # reciprocal of the distance to cluster center k by the sum of the reciprocals of the distances to each
        # cluster center from point i
        reciprocal_sum = 0
        for j in range(len(distances[i])):
            reciprocal_sum += 1 / (distances[i][j])
        for k in range(len(distances[i])):
            weights[i][k] = 1 / (distances[i][k] * reciprocal_sum)
    return weights

# Option A, Rule 2
# Method 2 - take harmonic weighted average of predictions of different models by distances to training clusters,
# no test clustering
def total_mse2(kmeans, num):
    # keeps track of the total squared error
    total = 0
    # compute the distances from each of the individual points in x_test to each cluster center of the training set
    distances = computeDistances(x_test, kmeans, num)
    # in this case, because all models will be used to make a prediction for each point in x_test, there is no need to
    # break the evaluation of the total squared error up into parts based on which specific model is used for prediction
    # this is why I assign y_true to be all of y_test
    y_true = y_test
    # the following is a numpy array of the same dimensions as distances
    # the ij entry of this array represents the prediction made for y at the ith point in x_test using model j
    predictions = np.zeros((len(x_test), num))
    for i in range(num):
        # create a Ridge object
        ridge = Ridge()
        # fit it only using data from the training set that is part of the training cluster with label i
        ridge.fit(x_train[kmeans.labels_ == i], y_train[kmeans.labels_ == i])
        # assign the ith column of predictions to be the set of predictions made at all points in x_test when model i
        # is used to make a prediction
        predictions[:,i] = ridge.predict(x_test)
    # for each point in the test set, the prediction made will be a weighted average of the predictions made for y at that point
    # using each model
    
    # calculate the weights that will need to be used
    weights = calculateWeights(distances)
    y_pred = np.zeros((len(y_true)))
    for j in range(len(predictions)):
        # this dot product computes the weighted average that will be used as the prediction at the jth point in x_test
        y_pred[j] = np.dot(weights[j], predictions[j])
    # add all squared errors to the total
    for j in range(len(y_true)):
        total += (np.linalg.norm(y_true[j] - y_pred[j]))**2
    # to get the mean squared error, divide the total by the number of data points
    return total / len(x_test)

# Option A, Rule 3
# Method 3 - take harmonic weighted average of predictions by squared distances to training clusters,
# no test clustering

# this function is exactly the same as the previous one, the only difference being that because a harmonic weighted
# average is being taken of the square distances instead of the distances, instead of passing distances into the calculateWeights
# function, one passes the componentwise square of distances (np.power(distances, 2)) into the calculateWeights function
def total_mse3(kmeans, num):
    total = 0
    distances = computeDistances(x_test, kmeans, num)
    y_true = y_test
    predictions = np.zeros((len(x_test), num))
    for i in range(num):
        ridge = Ridge()
        ridge.fit(x_train[kmeans.labels_ == i], y_train[kmeans.labels_ == i])
        predictions[:,i] = ridge.predict(x_test)
    weights = calculateWeights(np.power(distances, 2))
    y_pred = np.zeros((len(y_true)))
    for j in range(len(predictions)):
        y_pred[j] = np.dot(weights[j], predictions[j])
    for j in range(len(y_true)):
        total += (np.linalg.norm(y_true[j] - y_pred[j]))**2
    return total / len(x_test)

# Option B, Rule 1 
# Method 4 - all weight on closest training cluster center, test clustering
# For simplicity, the number of clusters in the test set is equal to the number in the training set
def total_mse4(kmeans, num):
    total = 0
    # cluster the test set, for simplicity with the same number of clusters as in the training set
    k_means_test = KMeans(num)
    k_means_test.fit(x_test)
    # this line computes, according to Rule 1, which model should be used to make a prediction at each
    # cluster center of the test set
    models = get_models(k_means_test.cluster_centers_, kmeans, num)
    for i in range(num):
        # create a Ridge object
        ridge = Ridge()
        # fit it only using training data in the cluster of the training set with label i
        ridge.fit(x_train[kmeans.labels_ == i], y_train[kmeans.labels_ == i])
        # trueIndices will represent the indices of the points in the test set that are part of a cluster in the test set such
        # that the output of models specifies that model i is to be used to predict y at the center of the relevant test cluster
        
        # because my method says to do the same thing at all points in a cluster of the test set as you would do at its cluster
        # center, trueIndices will specify the indices of whichever points in the test set should have y predicted using model i
        trueIndices = []
        for j in range(len(x_test)):
            if models[k_means_test.labels_[j]] == i:
                trueIndices.append(j)
        # if there are points in the test set such that y is predicted using model i, then add the squared errors in these
        # predictions to the total
        if len(trueIndices) > 0:
            y_pred = ridge.predict(x_test[np.array(trueIndices)])
            y_true = y_test[np.array(trueIndices)]
            for k in range(len(y_pred)):
                total += (y_pred[k] - y_true[k])**2
    return total / len(x_test)

# Option B, Rule 2
# Method 5 - take harmonic weighted average of predictions by distances to training clusters,
# test clustering
def total_mse5(kmeans, num):
    total = 0
    k_means_test = KMeans(num)
    # cluster the test set, for simplicity with the same number of clusters as in the training set
    k_means_test.fit(x_test)
    
    # compute the distances from each cluster center of the test set to each cluster center of the training set
    
    # the ij entry of distances represents the distance from cluster center i of the test set to cluster center j of
    # the training set
    distances = computeDistances(k_means_test.cluster_centers_, kmeans, num)
    # the true values of y
    y_true = y_test
    # this numpy array will hold the final predictions made for y for each point in the test set
    y_pred = np.zeros((len(y_test),))
    # in this predictions numpy array, it will be built so that the ji entry represents the prediction made for the
    # jth test data point by the ith model
    predictions = np.zeros((len(x_test), num))
    for i in range(num):
        # create a Ridge object
        ridge = Ridge()
        # train it only using data from the cluster of the training set with label i
        ridge.fit(x_train[kmeans.labels_ == i], y_train[kmeans.labels_ == i])
        # the ith column of predictions is the predictions that would be made for y for all points 
        # in the test set using model i
        predictions[:,i] = ridge.predict(x_test)
    # this calculates the weights that would be used to apply Rule 2 to the prediction of y at each cluster center
    # of the test set, and the jth row of weights is the weights calculated for the jth cluster center
    weights = calculateWeights(distances)
    # again, Option B says to use the same prediction rule at any point in a given cluster of the test set as one would at
    # the center of the relevant cluster, so if a weighted average with certain coefficients would be used to predict y at
    # the center of a certain cluster, a weighted average with the same coefficients will be used at each point in that cluster
    for j in range(num):
        # the indexing k_means_test.labels_ == j specifies precisely those indices of those points in x_test that are part of
        # the jth cluster of x_test
        
        # for each of those points in x_test, the final prediction for y will be a weighted average of the predictions each
        # model makes for y at that point, with the weights being those corresponding to cluster center j
        
        # for each individual point in x_test that is part of the jth cluster, the weighted average can be computed with a dot
        # product, but matrix multiplication can be used to compute the dot products for all of the relevant points in x_test at
        # once
        y_pred[k_means_test.labels_ == j] = np.matmul(predictions[k_means_test.labels_ == j], weights[j])
    # add all of the squared errors together
    for k in range(len(y_true)):
        total += (np.linalg.norm(y_true[k] - y_pred[k]))**2
    # compute the mean squared error
    return total / len(x_test)

# Option B, Rule 3
# Method 6 - take harmonic weighted average of predictions by squared distances to training clusters,
# test clustering

# this function is exactly the same as the previous one, the only difference being that because a harmonic weighted
# average is being taken of the square distances instead of the distances, instead of passing distances into the calculateWeights
# function, one passes the componentwise square of distances (np.power(distances, 2)) into the calculateWeights function.
def total_mse6(kmeans, num):
    total = 0
    k_means_test = KMeans(num)
    k_means_test.fit(x_test)
    distances = computeDistances(k_means_test.cluster_centers_, kmeans, num)
    y_true = y_test
    y_pred = np.zeros((len(y_test),))
    predictions = np.zeros((len(x_test), num))
    for i in range(num):
        ridge = Ridge()
        ridge.fit(x_train[kmeans.labels_ == i], y_train[kmeans.labels_ == i])
        predictions[:,i] = ridge.predict(x_test)
    weights = calculateWeights(np.power(distances, 2))
    for j in range(num):
        y_pred[k_means_test.labels_ == j] = np.matmul(predictions[k_means_test.labels_ == j], weights[j])
    for k in range(len(y_true)):
        total += (np.linalg.norm(y_true[k] - y_pred[k]))**2
    return total / len(x_test)       

# this next section of code calculates the mean squared error using varying number of clusters from 1 to 10
# of each of the six approaches and puts the calculations in a 10 x 6 matrix   
meanSquaredErrors = np.zeros((10, 6))
for k in range(1, 11):
    kmeans = KMeans(k)
    kmeans.fit(x_train)
    meanSquaredErrors[k - 1][0] = total_mse1(kmeans, k)
    meanSquaredErrors[k - 1][1] = total_mse2(kmeans, k)
    meanSquaredErrors[k - 1][2] = total_mse3(kmeans, k)
    meanSquaredErrors[k - 1][3] = total_mse4(kmeans, k)
    meanSquaredErrors[k - 1][4] = total_mse5(kmeans, k)
    meanSquaredErrors[k - 1][5] = total_mse6(kmeans, k)

# a plot comparing Rules 1, 2, and 3, holding the usage of Option A fixed    
plt.figure()
plt.title("Mean squared error versus number of clusters in training set, Option A")
plt.xlabel("Number of clusters in training set")
plt.ylabel("Mean squared error")
plt.plot(range(1, 11), meanSquaredErrors[:, 0], label = "Rule 1")
plt.plot(range(1, 11), meanSquaredErrors[:, 1], label = "Rule 2")
plt.plot(range(1, 11), meanSquaredErrors[:, 2], label = "Rule 3")
plt.legend()

# a plot comparing Rules 1, 2, and 3, holding the usage of Option B fixed
plt.figure()
plt.title("Mean squared error versus number of clusters in training set, Option B")
plt.xlabel("Number of clusters in training set")
plt.ylabel("Mean squared error")
plt.plot(range(1, 11), meanSquaredErrors[:, 3], color = 'red', label = "Rule 1")
plt.plot(range(1, 11), meanSquaredErrors[:, 4], color = 'magenta', label = "Rule 2")
plt.plot(range(1, 11), meanSquaredErrors[:, 5], color = 'brown', label = "Rule 3")
plt.legend()

# a plot comparing Options A and B, holding the usage of Rule 1 fixed
plt.figure()
plt.title("Mean squared error versus number of clusters in training set, Rule 1")
plt.xlabel("Number of clusters in training set")
plt.ylabel("Mean squared error")
plt.plot(range(1, 11), meanSquaredErrors[:, 0], label = "Option A")
plt.plot(range(1, 11), meanSquaredErrors[:, 3], label = "Option B")
plt.legend()

# a plot comparing Options A and B, holding the usage of Rule 2 fixed
plt.figure()
plt.title("Mean squared error versus number of clusters in training set, Rule 2")
plt.xlabel("Number of clusters in training set")
plt.ylabel("Mean squared error")
plt.plot(range(1, 11), meanSquaredErrors[:, 1], color = "orange", label = "Option A")
plt.plot(range(1, 11), meanSquaredErrors[:, 4], color = "red", label = "Option B")
plt.legend()

# a plot comparing Options A and B, holding the usage of Rule 3 fixed
plt.figure()
plt.title("Mean squared error versus number of clusters in training set, Rule 3")
plt.xlabel("Number of clusters in training set")
plt.ylabel("Mean squared error")
plt.plot(range(1, 11), meanSquaredErrors[:, 2], color = "magenta", label = "Option A")
plt.plot(range(1, 11), meanSquaredErrors[:, 5], color = "brown", label = "Option B")
plt.legend()

# a plot comparing all six methods at once (the results of all six methods plotted on the same figure)
plt.figure()
plt.title("Mean squared error versus number of clusters in the training set, all methods")
plt.plot(range(1, 11), meanSquaredErrors[:,0], label = "Option A, Rule 1")
plt.plot(range(1, 11), meanSquaredErrors[:, 1], label = "Option A, Rule 2")
plt.plot(range(1, 11), meanSquaredErrors[:, 2], label = "Option A, Rule 3")
plt.plot(range(1, 11), meanSquaredErrors[:, 3], label = "Option B, Rule 1")
plt.plot(range(1, 11), meanSquaredErrors[:, 4], label = "Option B, Rule 2")
plt.plot(range(1, 11), meanSquaredErrors[:, 5], label = "Option B, Rule 3")
plt.title("Mean squared error versus number of clusters used in training set")
plt.xlabel("Number of clusters used in training set")
plt.ylabel("Mean squared error")
plt.legend()
plt.show()
