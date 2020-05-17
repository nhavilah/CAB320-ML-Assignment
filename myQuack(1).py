
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
import sys
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical


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
    model = tree.DecisionTreeClassifier()
    # adjust arange values for the values tested
    params = {"max_depth": np.arange(1, 15, 1)}
    clf = GridSearchCV(model, params, cv=10)
    clf.fit(X_training, y_training)
    print("[INFO] randomized search best parameters: {}".format(clf.best_params_))

# clf = tree.DecisionTreeClassifier(max_depth=6, random_state=100)
# clf = clf.fit(X_training, y_training)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jobs", type=int, default=-1,
                    help="# of jobs for k-NN distance (-1 uses all available cores)")
    args = vars(ap.parse_args())
    model = KNeighborsClassifier(n_jobs=args["jobs"])
    # adjust arange values for the values tested
    params = {"n_neighbors": np.arange(1, 15, 1)}
    clf = GridSearchCV(model, params, cv=10)
    clf.fit(X_training, y_training)
    print("[INFO] randomized search best parameters: {}".format(clf.best_params_))
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
    model = svm.SVC()
    # adjust arange values for the values tested
    params = {"C": np.arange(1, 15, 1)}
    clf = GridSearchCV(model, params, cv=10)
    clf.fit(X_training, y_training)
    print("[INFO] randomized search best parameters: {}".format(clf.best_params_))
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

    hidden_layer_sizes = [5, 10]
    # hidden_layer_sizes = to_categorical(hidden_layer_sizes, num_classes = 2)

    # hidden_layer_sizes = [5, 10]

    params = dict(hidden_layer_sizes=hidden_layer_sizes)

    # model = KerasClassifier(build_fn = DL_Model, epochs = 50, batch_size = 40, verbose = 0)
    model = KerasClassifier(build_fn=DL_Model, epochs=10,
                            batch_size=40, verbose=0)

    clf = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=10)

    clf.fit(X_training, y_training)

    print("[INFO] randomized search best parameters: {}".format(clf.best_params_))
    return clf

    # model = MLPClassifier()
    # # adjust arange values for the values tested
    # params = [
    #     {
    #         "hidden_layer_sizes": [(40,)]
    #     }
    # ]

    # clf = GridSearchCV(model, params, cv=10)
    # clf.fit(X_training, y_training)
    # print("[INFO] randomized search best parameters: {}".format(clf.best_params_))
    # return clf


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


def DL_Model(hidden_layer_sizes=5):
    model = Sequential()
    # model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(hidden_layer_sizes, input_dim=8))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_layer_sizes))
    model.add(Dropout(0.5))
    model.compile(loss='categorical_crossentropy', optimizer='adam', activation='softmac', metrics=['accuracy'])
    return model
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

    # clf = build_DecisionTree_classifier(
    #     x_train, y_train)  # decision tree classifier

    # clf = build_NearrestNeighbours_classifier(
    #     x_train, y_train)  # nearest neighbours classifier

    # clf = build_SupportVectorMachine_classifier(
    #     x_train, y_train)  # svm classifier

    clf = build_NeuralNetwork_classifier(
        x_train, y_train)  # neural network classifier

    # call the methods that will evaluate classifier performance
    check_Classifier_Training_Performance(clf, x_train, y_train)
    check_Classifier_Testing_Performance(clf, x_test, y_test)
    classifier_Performance_Report(clf, x_test, y_test)
    classifier_Confusion_Matrix(clf, x_test, y_test)

    # print the performance data
    print(clf)
