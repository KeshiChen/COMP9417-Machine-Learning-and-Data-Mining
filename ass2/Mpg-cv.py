import time
import pandas as pd
#from scipy.io import arff
from MLPerceptron import *
import numpy as np

def label_race(row, col_name, value):
    #print(int(row[col_name]))
    if float(row[col_name]) == value:
        #print('aaaa')
        return 1
    else:
        return 0
def split_train_test(x, y, test_size):
    x = np.asarray(x, dtype=np.float64)
    test_num = int(x.shape[0] * test_size)
    test_idx = np.random.choice(x.shape[0], test_num).tolist()
    test_x = x[test_idx]
    test_y = y[test_idx]
    train_x = np.delete(x, test_idx, axis=0)
    train_y = np.delete(y, test_idx)
    return train_x, test_x, train_y, test_y

def preprocessing(filename):
    from scipy.io import arff
    data = arff.loadarff(filename)
    column_names = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model', 'origin', 'class']
    df = pd.DataFrame(data[0], columns=column_names)
    for i in df.columns:
        df[i] = df[i].fillna(df[i].mean())
    Cylinders = [8, 4, 6, 3, 5]
    for value in Cylinders:
        df['cylinder_' + str(value)] = df.apply(lambda row: label_race(row, 'cylinders', value), axis=1)
    Model = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]
    for value in Model:
        df['model_' + str(value)] = df.apply(lambda row: label_race(row, 'model', value), axis=1)
    Origin = [1, 2, 3]
    for value in Origin:
        df['origin_' + str(value)] = df.apply(lambda row: label_race(row, 'origin', value), axis=1)
    df = df.drop(columns=['cylinders', 'model', 'origin'], axis=1)
    y = np.asarray(df['class'], dtype=np.float64)
    df = df.drop(columns=['class'])
    x = np.asarray(df.loc[:, 'displacement':'origin_3'], dtype=np.float64)
    return x,y

filename = "autoMpg.arff"
x,y = preprocessing(filename)
n_iter = 20000
eta = 0.01
pp = MLPerceptron((25,  1), n_iter=n_iter, rate=eta, activator='linear')


def StandardScaler_Fit(X):
    mean = []
    std = []
    #print(X.iloc[:, 0])
    for i in range(X.shape[1]):
        mean.append(np.mean(X[:, i]))
        std.append(np.std(X[:, i]))
    return mean, std


def StandardScaler_Transform(X, mean, std):
    X_copy = X.copy()
    m = np.array(mean)
    s = np.array(std)
    #print(X_copy[0,:])
    for i in range(X.shape[0]):
        X_copy[i,:] = (X_copy[i,:] - m) / s
    return X_copy

def exp_var(y_true,y_target):
    diff=[]
    for i in range(len(y_true)):
        diff.append(y_true[i]-y_target[i])
    return 1-(np.var(diff)/np.var(y_true))

def cross_validation(k,n,x,y):
    accuracy=[]
    for i in range(n):
        x_train, x_test, y_train, y_test = split_train_test(x, y, test_size=1/k)
        mean, std = StandardScaler_Fit(x_train)
        X_train_std = StandardScaler_Transform(x_train, mean, std)
        X_train_std = np.array(X_train_std, dtype=np.float64)
        X_test_std = StandardScaler_Transform(x_test, mean, std)
        X_test_std = np.array(X_test_std, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float)
        pp.train(X_train_std, y_train)
        y_predicted = pp.predict(X_test_std)
        y_test = np.asarray(y_test, dtype=np.float)
        score = exp_var(y_test,y_predicted)
        accuracy.append(score)
    return np.mean(accuracy)


cross_validation(10,20,x,y)
