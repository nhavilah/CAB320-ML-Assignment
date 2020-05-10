
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [(10469231, 'Nicholas', 'Havilah'), (10522662, 'Connor', 'McHugh'), (1234569, 'Eva', 'Tardos')]
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
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_training, y_training)
    # can uncomment this for reporting purposes-shows the full extent of the decision tree
    # plt.figure()
    # cn = ['M', 'B']
    # tree.plot_tree(clf, filled=True, class_names=cn)
    # plt.show()
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
    clf = KNeighborsClassifier(n_neighbors=4)
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
    raise NotImplementedError()

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
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
    # "INSERT YOUR CODE HERE"
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
    pass
    # Write a main part that calls the different
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    # prepare the genralised datasets
    x, y = prepare_dataset('D:/dOWNLOADS/medical_records(1).data')
    # define training and test data to be used for accuracy measurement
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # create the classifiers
    # decision tree classifier
    dtc = build_DecisionTree_classifier(x_train, y_train)
    # nearest neighbor classifier
    nnc = build_NearrestNeighbours_classifier(x_train, y_train)

    # test how accurate the classifiers are using Kfold cross validation
    # test the decision tree
    dtc_cv_scores = cross_val_score(dtc, x_test, y_test, cv=10)
    # print the average result to get a better idea of general performance with the test set
    print(np.mean(dtc_cv_scores))
