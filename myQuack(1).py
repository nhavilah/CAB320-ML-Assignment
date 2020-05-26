
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
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
    params = {"max_depth": np.arange(1, 10, 1)}

    # clf = GridSearchCV(model, params, cv=10, n_jobs=-1, scoring=[
    #                    'accuracy', 'precision', 'roc_auc', 'recall', 'f1'], refit='accuracy')
    clf = GridSearchCV(model, params, cv=10, n_jobs=-1,
                       scoring=['accuracy'], refit='accuracy')
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
    params = {"n_neighbors": np.arange(8, 30, 1)}

    # clf = GridSearchCV(model, params, cv=10, scoring=[
    #                    'accuracy', 'precision', 'roc_auc', 'recall', 'f1'], refit='accuracy')
    clf = GridSearchCV(model, params, cv=10, n_jobs=-1,
                       scoring=['accuracy'], refit='accuracy')
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
    model = svm.SVC(random_state=1)
    # adjust arange values for the values tested
    params = {"C": np.arange(1, 15, 1)}
    # clf = GridSearchCV(model, params, cv=10, n_jobs=-1, scoring=[
    #                    'accuracy', 'precision', 'roc_auc', 'recall', 'f1'], refit='accuracy')

    clf = GridSearchCV(model, params, cv=10, n_jobs=-1,
                       scoring=['accuracy'], refit='accuracy')
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
    model = MLPClassifier(max_iter=1500, random_state=10)
    params = {'hidden_layer_sizes': np.arange(60, 70, 1)}

    # clf = GridSearchCV(model, params, cv=10, n_jobs=-1,
    #                    scoring=['accuracy', 'precision', 'roc_auc', 'recall', 'f1'], refit='accuracy')
    clf = GridSearchCV(model, params, cv=10, n_jobs=-1,
                       scoring=['accuracy'], refit='accuracy')
    clf.fit(X_training, y_training)
    return clf
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
# "INSERT YOUR CODE HERE"

# Plots a graph of


def plot_Hyperparameter_Op(clf, x_test, y_test):
    # Create and array of the hyperparameter that was tuned
    parameters = list()
    for parameter in clf.param_grid.keys():
        parameters.append(parameter)

    # Create an array of possible values for the hyperparameter
    values = list()
    for value in clf.param_grid.values():
        values.append(value)
    values = list(values)[0]

    metrics = clf.scoring

    # Create an array of all of the scores
    scores = list()
    for metric in metrics:
        scores.append(clf.cv_results_['mean_test_%s' % metric])

    metric_num = 0
    for score in scores:
        plt.scatter(values, score)
        z = np.polyfit(values, score, 4)
        p = np.poly1d(z)
        plt.plot(values, p(values), "--", label=metrics[metric_num])
        metric_num += 1

    plt.xlabel(parameters[0])
    plt.ylabel('average test score')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('plots/' + parameters[0])
    plt.show()


# Gets the testing and training accuracy for the provided classifier
def check_Classifier_Both_Performance(clf):
    test_score = clf.score(x_test, y_test) * 100
    train_score = clf.score(x_train, y_train) * 100
    test_error = 100 - test_score
    train_error = 100 - train_score
    print('Train Accuracy:\t%.2f%%' % train_score,
          '\t\tTrain Error:\t%.2f%%' % train_error)
    print('Test Accuracy:\t%.2f%%' % test_score,
          '\t\tTest Error:\t%.2f%%' % test_error)
    
    
# evaluates the classifier's accuracy using the testing data
def check_Classifier_Testing_Performance(clf, x_testing, y_testing):
    results = balanced_accuracy_score(
        clf.predict(x_testing), y_testing).mean()*100
    print("Accuracy: %.2f%%" % results)


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
    # print("Confusion Matrix:")
    # print(results)
    plt.show()

# creates a plot showing accuracy of the classifier using two key features
# code sourced from https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py


def plot_Classifier_Performance(clf, x_testing, y_testing):
    plt.figure()
    reduced_data = x_testing[:, :2]
    n_classes = 2
    plot_colors = "rb"
    plot_step = 0.02
    clf.fit(reduced_data, y_testing)
    h = 0.02
    x_min, x_max = reduced_data[:, 0].min()-1, reduced_data[:, 0].max()+1
    y_min, y_max = reduced_data[:, 1].min()-1, reduced_data[:, 0].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    plt.show()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
    # prepare the genralised datasets
    x, y = prepare_dataset('./medical_records.data')

    # define training and test data to be used for accuracy measurement
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=100)



# NOTE TO THE ASSESSOR:
# Please uncomment the classifier that you would like to test, as well as
# any testing / plotting funtions that you wish to use


# Classifiers

#-------- Decision Tree Classifier ---------------------------------------#
    # print("Decision Tree")
    # clf = build_DecisionTree_classifier(
    #     x_train, y_train)
#-------------------------------------------------------------------------#


#-------- Nearest Neighbour Classifier -----------------------------------#
    # print("Nearest Neighbours")
    # clf = build_NearrestNeighbours_classifier(
    #     x_train, y_train)
#-------------------------------------------------------------------------#


#-------- SVM Classifier -------------------------------------------------#
    print("SVM")
    clf = build_SupportVectorMachine_classifier(
        x_train, y_train)
#-------------------------------------------------------------------------#


#-------- Neural Network Classifier --------------------------------------#
    # print("Neural Network")
    # clf = build_NeuralNetwork_classifier(
    #     x_train, y_train)
#-------------------------------------------------------------------------#


# Classifier performance / plotting methods


#-------- Print testing / training scores --------------------------------#
    check_Classifier_Both_Performance(clf)
#-------------------------------------------------------------------------#


#-------- Print the accuracy percentage ----------------------------------#
    # check_Classifier_Testing_Performance(clf, x_test, y_test)
#-------------------------------------------------------------------------#


#-------- Print the best parameters --------------------------------------#
    print("Best parameters: {}".format(clf.best_params_))
#-------------------------------------------------------------------------#


#-------- Print Classifier Performance Report-----------------------------#
    classifier_Performance_Report(clf, x_test, y_test)
#-------------------------------------------------------------------------#


#-------- Plot classifier confusion matrix--------------------------------#
    classifier_Confusion_Matrix(clf, x_test, y_test)
#-------------------------------------------------------------------------#


#-------- Plot hyperparameter graph --------------------------------------#
    plot_Hyperparameter_Op(clf, x_test, y_test)
#-------------------------------------------------------------------------#


# plot_Classifier_Performance(clf, x_test, y_test)
