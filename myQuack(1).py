
'''

2020

Scaffolding code for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes as necessary.

You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


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
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X_training, y_training)
    # return clf

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
    raise NotImplementedError()

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

    # "INSERT YOUR CODE HERE"
    # prepare the genralised datasets
    x, y = prepare_dataset('D:\dOWNLOADS\medical_records(1).data')
    # prepare 80% of the dataset as training data
    # actual values

    # classification data

    # prepare 20% of the dataset as test data
    # actual values

    # classification data
