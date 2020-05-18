
'''
2020
Scaffolding code for the Machine Learning assignment.
You should complete the provided functions and add more functions and classes as necessary.
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.
You are welcome to use the pandas library if you know it.
'''
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.constraints import maxnorm
import time


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    '''
    return [(10469231, 'Nicholas', 'Havilah'), (10522662, 'Connor', 'McHugh'), (9448977, 'Kevin', 'Duong')]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''
    Read a comma separated text file where
        - the first field is a ID number
        - the second field is a class label 'B' or 'M'
        - the remaining fields are real-valued
    Return two numpy arrays X and y where
        - X is two dimensional. X[i,:] is the ith example
        - y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'
    @param dataset_path: full path of the dataset text file
    @return
        X,y
    '''
    # "INSERT YOUR CODE HERE"
    x = np.genfromtxt(dataset_path, delimiter=',')
    yList = []
    ynumpy = np.genfromtxt(dataset_path, delimiter=',',
                           dtype=None, encoding=None)
    for row in range(ynumpy.shape[0]):
        currentRow = ynumpy[row]
        if currentRow[1] == 'M':
            yList.append(1)
        else:
            yList.append(0)
    y = np.array(yList)
    x = np.delete(x, 0, axis=1)
    x = np.delete(x, 0, axis=1)
    return (x, y)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_DecisionTree_classifier(X_training, y_training):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.
    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
    @return
        clf : the classifier built in this function
    '''
    # "INSERT YOUR CODE HERE"
    # note that the max depth is the variable you want to play around with to get the best possible classifier
    model = tree.DecisionTreeClassifier(random_state=1)
    # adjust arange values for the values tested
    params = {"max_depth": np.arange(1, 30, 1)}
    clf = GridSearchCV(model, params, cv=10, n_jobs=-1)
    clf.fit(X_training, y_training)
    return clf
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_NearrestNeighbours_classifier(X_training, y_training):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.
    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
    @return
        clf : the classifier built in this function
    '''
    # "INSERT YOUR CODE HERE"
    # play around with this hyperparameter to get accuracy as close as possible
    model = KNeighborsClassifier()
    # adjust arange values for the values tested
    params = {"n_neighbors": np.arange(1, 15, 1)}
    clf = GridSearchCV(model, params, cv=10)
    clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_SupportVectorMachine_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.
    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
    @return
        clf : the classifier built in this function
    '''
    # "INSERT YOUR CODE HERE"
    model = svm.SVC(random_state=100)
    # adjust arange values for the values tested
    params = {"C": np.arange(1, 15, 1)}
    clf = GridSearchCV(model, params, cv=10)
    clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_NeuralNetwork_classifier(X_training, y_training):
    '''
    Build a Neural Network classifier (with two dense hidden layers)
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library
    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
    @return
        clf : the classifier built in this function
    '''
    model = MLPClassifier(max_iter=1500, random_state=100)
    params = {'hidden_layer_sizes': np.arange(60, 70, 1)}
    clf = GridSearchCV(model, params, cv=10, n_jobs=-1)
    clf.fit(X_training, y_training)
    return clf
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
# "INSERT YOUR CODE HERE"

# evaluates the classifier's accuracy using the training data(we need this to show we aren't overfitting the data)


def check_Classifier_Training_Performance(clf, x_training, y_training):
    results = cross_val_score(clf, x_training, y_training, cv=10).mean()*100
    print("Training Prediction Accuracy: %.2f%%" % results)

# evaluates the classifier's accuracy using the testing data


def check_Classifier_Testing_Performance(clf, x_testing, y_testing):
    results = cross_val_score(clf, x_testing, y_testing, cv=10).mean()*100
    print("Testing Prediction Accuracy: %.2f%%" % results)
    print("Best parameters: {}".format(clf.best_params_))

# creates a report on the classifier that we can use to show how accurate it is


def classifier_Performance_Report(clf, x_testing, y_testing):
    prediction = clf.predict(x_testing)
    results = classification_report(prediction, y_testing)
    print("Report:")
    print(results)

# creates a confusion matrix that allows us to see more of what's happening with the data after the classifier operates on it


def classifier_Confusion_Matrix(clf, x_testing, y_testing):
    results = plot_confusion_matrix(
        clf, x_testing, y_testing, normalize='true')
    print("Confusion Matrix:")
    print(results)
    plt.show()

# creates a plot showing accuracy of the classifier using two key features
# code sourced from https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py


def plot_Classifier_Performance(clf, x_testing, y_testing, x_training, y_training):
    plt.figure()
    reduced_data = x_testing[:, :2]
    reduced_data2 = x_training[:, :2]
    h = 0.02
    x_min, x_max = reduced_data[:, 0].min()-1, reduced_data[:, 0].max()+1
    y_min, y_max = reduced_data[:, 1].min()-1, reduced_data[:, 0].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.scatter(reduced_data2, y_training)
    plt.show()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
    # Write a main part that calls the different
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    # prepare the genralised datasets
    x, y = prepare_dataset('./medical_records.data')
    # define training and test data to be used for accuracy measurement
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=100)
    # for assessor purposes, uncomment the classifier you want to use
    # no other lines should need to be commented out, as the rest of the code
    # handles training and testing for you
    # list of classifiers

    start_time = time.time()

    clf = build_DecisionTree_classifier(
        x_train, y_train)  # decision tree classifier

    # clf = build_NearrestNeighbours_classifier(
    #     x_train, y_train)  # nearest neighbours classifier

    # clf = build_SupportVectorMachine_classifier(
    #     x_train, y_train)  # svm classifier

    # clf = build_NeuralNetwork_classifier(
    #     x_train, y_train)  # neural network classifier

    # call the methods that will evaluate classifier performance
    check_Classifier_Training_Performance(clf, x_train, y_train)
    check_Classifier_Testing_Performance(clf, x_test, y_test)
    # classifier_Performance_Report(clf, x_test, y_test)

    # classifier_Confusion_Matrix(clf, x_test, y_test)
    plot_Classifier_Performance(clf, x_test, y_test, x_train, y_train)
    # print the performance data
    # print(clf)
    run_time = time.time() - start_time
    print('The program took ', run_time, ' seconds to run')
