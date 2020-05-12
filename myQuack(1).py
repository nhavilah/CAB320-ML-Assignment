
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
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [(10469231, 'Nicholas', 'Havilah'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos')]
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
    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=0)
    clf = clf.fit(X_training, y_training)
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
    clf = KNeighborsClassifier(n_neighbors=1, random_state=0)
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
    clf = svm.SVC(C=1, random_state=0)
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
    # "INSERT YOUR CODE HERE"
    clf = MLPClassifier(random_state=0)
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
# "INSERT YOUR CODE HERE"
# here will be the method that validates how good the classifiers are
def check_Classifier_Performance(X_test, y_test, clf):
    results = clf.predict(X_test)
    average_results = accuracy_score(y_test, results)
    return average_results*100

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
    pass
    # Write a main part that calls the different
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    # prepare the genralised datasets
    x, y = prepare_dataset('D:/dOWNLOADS/medical_records(1).data')
    # define training and test data to be used for accuracy measurement
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

    # for assessor purposes, uncomment the classifier you want to use
    # no other lines should need to be commented out, as the rest of the code
    # handles training and testing for you
    # list of classifiers

    clf = build_DecisionTree_classifier(
        x_train, y_train)  # decision tree classifier

    # clf = build_NearrestNeighbours_classifier(
    #     x_train, y_train)  # nearest neighbours classifier

    # clf = build_SupportVectorMachine_classifier(
    #     x_train, y_train)  # svm classifier

    # clf = build_NeuralNetwork_classifier(
    #     x_train, y_train)  # neural network classifier

    # call the test method
    performance_measure = check_Classifier_Performance(x_test, y_test, clf)
    print("Accuracy: %.2f%%" % performance_measure)
