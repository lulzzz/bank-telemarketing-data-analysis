import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sys


def dataset():
    data = pd.read_csv('train.csv')
    data_dummies = pd.get_dummies(data)

    features = data_dummies.ix[:, 'age':'poutcome_success']

    X_train = features.values
    y_train = data_dummies['y_yes'].values

    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)

    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    return (X_train, y_train)


def testset():
    data = pd.read_csv('test.csv')
    data_dummies = pd.get_dummies(data)

    features = data_dummies.ix[:, 'age':'poutcome_success']
    X_test = features.values

    scaler = preprocessing.MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    return X_test


if __name__ == '__main__':
    (X_train, y_train) = dataset()
    X_test = testset()

    ###########################################
    # Choose SVM() of RandomForestClassifier()
    ###########################################
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)

    print(y_test)

    # Csv write
    filename = 'submission.csv'
    with open(filename, 'w') as f:
        f.write('Id,prediction\n')
        for num in range(1, len(y_test) + 1):
            f.write(str(num) + ',' + str(y_test[num - 1]) + '\n')

        print('complete!')
