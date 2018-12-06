import numpy as np

class MyPerceptron:

    def activate_perceptron(x):
        if x > 0:
            return 1
        else:
            return 0


    def __init__(self, n_iter= -1, eta= 1.0, converge_threshold= 0.001, activator= activate_perceptron):
        self.n_iter = n_iter
        self.eta = eta
        self.w = np.array([0])
        self.b = 0.0
        self.converge_threshold = converge_threshold
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_predicted = None
        self.activator = activator
        self.prev_delta = np.inf

    def __check__(self):
        for idx in range(self.x_train.shape[0]):
            if idx != 0:
                print('not 0')
            xi = self.x_train[idx,:]
            #y = np.dot(self.w.T, xi) + self.b
            y = self.__predict__(xi)
            #print('xi',xi)
            Y = self.y_train[idx]
            delta = Y - y
            #print('check',Y, y, delta)
            d = abs(delta)
            if d > self.converge_threshold:
                #print('return')
                #print('d',d, self.converge_threshold)
                self.prev_delta = delta
                return xi, delta
        print('converged!')
        return 'converged', 'concerged'

    def gradient_decent(self, y_true, y_pred, x):
        d_Y = y_true - y_pred
        #print(d_Y.shape[0])
        #print(x.shape[1])
        sum = np.array([0.0 for i in range(x.shape[1])])
        for i in range(d_Y.shape[0]):
            #print('y-y, xi', y_true[i] - y_pred[i], x[i])
            if not np.isnan(y_true[i] - y_pred[i]):
                sum += (y_true[i] - y_pred[i]) * x[i]
            else:
                sum += 0.0
        #print('sum', sum)
        return sum

    def __update__(self, xi, delta):
        x = np.array(xi)
        w = np.array(self.w)
        b = self.b
        #print('delta', delta)
        #print(xi)
        self.w += self.eta * delta * xi
        self.b += self.eta * delta
        #print('wb',self.w, self.b)
        # w = np.append(w, b)
        # w1 = np.append(self.w, self.b)
        return 0
        # if abs(la.norm(w1, 2) - la.norm(w, 2)) <= self.converge_threshold:
        #     return 1
        # else:
        #     return 0

    import numpy as np
    import random


    def train(self, x_train, y_train, activator = activate_perceptron):
        #print('shape', x_train.shape)
        self.class_weights = {}
        self.activator = activator
        self.w = np.array([0.0 for i in range(x_train.shape[1])])
        self.b = 0.0
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.y_predicted = np.array([0.0 for i in range(x_train.shape[0])])
        for i in range(self.n_iter):
            #self.y_predicted = self.predict(self.x_train)
            # Batch Gradient Descent
            #self.b = self.b + self.eta *
            #print('prev, w',self.prev_weight, self.w)
            # if np.abs(np.average(self.w - self.prev_delta)) <= self.converge_threshold:
            #     print('converge')
            #     return
            xi, delta = self.__check__()
            if isinstance(xi, str):
                print('converged')
                return
            #print('xi', xi.shape)
            self.__update__(xi, delta)
        return

    def __predict__(self, sample):
        y = np.dot(self.w.T, sample) + self.b
        return self.activator(y)

    def predict(self, x_test):
        self.x_test = np.array(x_test)
        y = []
        for idx, sample in enumerate(self.x_test):
            y.append(self.__predict__(sample))
        return np.array(y)