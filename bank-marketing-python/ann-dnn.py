import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import layers, models
from keras.utils import np_utils


class ANN(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


class DNN(models.Sequential):
    def __init__(self, Nin, Nh_l, Pd_l, Nout):
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu',
                              input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dropout(Pd_l[0]))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dropout(Pd_l[1]))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])


def dataset():
    data = pd.read_csv('train.csv')
    data_dummies = pd.get_dummies(data)

    features = data_dummies.ix[:, 'age':'poutcome_success']
    X_train = features.values
    y_train = data_dummies['y_yes'].values

    # UnderSampling
    # rus = RandomUnderSampler(return_indices=True)
    # X, y, idx_resampled = rus.fit_sample(X, y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)

    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    return X_train, y_train


def testset():
    data = pd.read_csv('test.csv')
    data_dummies = pd.get_dummies(data)

    features = data_dummies.ix[:, 'age':'poutcome_success']
    X_test = features.values

    scaler = preprocessing.MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    return X_test


def plot_acc(history, title=None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Verification'], loc=0)
    plt.show()


def plot_loss(history, title=None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Verification'], loc=0)
    plt.show()


if __name__ == '__main__':
    # Data Extraction
    (X_train, y_train) = dataset()
    X_test = testset()
    y_test = None

    # Common Variable Setting
    Nin = X_train.shape[1]
    Nh = 10
    Nout = y_train.shape[1]
    # DNN Variable Setting
    Nh_l = [20, 10]  # 2 Layers
    Pd_l = [0.4, 0.4]  # Dropout

    ##############################
    # Choose ANN or DNN
    ##############################
    # model = ANN(Nin, Nh, Nout)
    model = DNN(Nin, Nh_l, Pd_l, Nout)

    # Train the model
    history = model.fit(X_train, y_train, epochs=100,
                        validation_split=0.01, verbose=1)

    # Model prediction
    y_test = model.predict_on_batch(X_test)

    # Csv write
    filename = 'submission.csv'
    with open(filename, 'w') as f:
        f.write('Id,prediction\n')
        for num in range(1, len(y_test) + 1):
            result = 0
            if y_test[num - 1][1] > 0.5:
                result = 1
            f.write(str(num) + ',' + str(result) + '\n')
        print('complete!')
