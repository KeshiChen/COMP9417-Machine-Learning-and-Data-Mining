import numpy as np
import random

class MLPerceptron:

    # Activations and Derivatives
    def linear(X): # used by linear unit
        return X

    def delinear(X):
        return 1.0

    def boolean(X): # used by perceptron to learn linearly-separable Boolean function
        if X > 0:
            return 1
        else:
            return 0

    def sigmoid(X): # used by MLP to learn non-linearly-separable Boolean function
        return 1.0 / (1 + np.exp(-X))

    def desigmoid(y):
        return y * (1 - y)

    # Initialize the class
    # Inputs: layers: tuple, each element indicates neuron number in corresponding layer, the first and the last element
    #                 means input and output layer, respectively
    #         activator: name of activation functions ('sigmoid', 'linear', 'boolean')
    #         n_iter: iteration times
    #         rate: learning rate
    #         use_mlp: whether to use multilayer perceptron, default is True, if False, use perceptron instead
    # Note: if user don't give hidden layer, e.g. layer= (25, 1), then the algorithm will be linear unit instead of MLP
    #       because there is no back-propagation, but only weight updates.
    def __init__(self, layers=(1,1,1), activator='sigmoid', n_iter=20, rate=0.01, use_mlp = True):
        self.n_iter = n_iter
        self.layers = []
        self.rate = rate
        self.weights = []
        self.weights0 = []
        self.use_mlp = use_mlp
        self.activator_name = activator
        if self.use_mlp:
            for i in range(len(layers) - 1):
                weight = np.log(np.random.random((layers[i], layers[i+1])))
                layer = np.ones(layers[i])
                self.weights.append(weight)
                self.layers.append(layer)
            layer = np.ones(layers[-1])
            self.layers.append(layer)
            for i in range(1, len(layers)):
                weight0 = np.random.random(layers[i])
                self.weights0.append(weight0)

        if activator == 'sigmoid':
            self.activation = MLPerceptron.sigmoid
            self.deactivation = MLPerceptron.desigmoid
        elif activator == 'linear':
            self.activation = MLPerceptron.linear
            self.deactivation = MLPerceptron.delinear
        elif activator == 'boolean':
            self.activation = MLPerceptron.boolean

    # Transform boolean output into the form of +1/-1
    def sign(self, y):
        if y > 0:
            return 1
        else:
            return -1

    # The implementation of pocket algorithm
    def pocket(self, X, y):
        # We randomly initiate weights
        Wpk, self.weight = np.random.rand(1, X.shape[1]+1), np.random.rand(1, X.shape[1]+1)
        # rpk: highest correct number of randomly selected samples
        # rpe: correct number of randomly selected samples
        # Pkt: highest correct number of all samples
        # Ppe: correct number of all samples
        rpk, rpe, Pkt, Ppe = 0, 0, 0, 0
        for i in range(self.n_iter):
            r = random.randint(0, X.shape[0]-1)
            if self.sign(y[r]) * self.sign(self._predict(X[r])) > 0:  # if correct
                rpe += 1
                if (rpe > rpk):
                    Y  = self.predict(X)
                    Ppe = len([y[i] for i in range(y.shape[0]) if y[i] == Y[i]])
                    if (Ppe > Pkt):
                        Wpk = self.weight.copy()
                        rpk = rpe
                        Pkt = Ppe
                        if (Pkt == X.shape[0]):
                            break
            # if not correct, update weights
            else:
                # update b
                self.weight[0, 0] = self.weight[0, 0] + self.rate * self.sign(y[r])
                # update w vector
                self.weight[0, 1:] = self.weight[0, 1:] + self.rate * self.sign(y[r]) * X[r]
                rpe = 0
        return Wpk

    # Train the model
    # Inputs: X: training samples
    #         y: labels of samples
    #         use_pocket: whether to use Pocket Algorithm, default True
    # Note: if user didn't set use_mlp to False in the initialization, then the training algorithm will use MLP instead.
    def train(self, X, y, use_pocket=True):
        # if user choose to use MLP or linear unit:
        if self.use_mlp:
            for _ in range(self.n_iter * (X.shape[0] // 1)):
                selected_idx = np.random.choice(X.shape[0], 1)
                # First of all, we feed forward from input layer to output layer
                self.forward(X[selected_idx])
                # Then we do back propagation to update weights from output layer to input layer
                self.back_propagate(y[selected_idx])
        # if user choose to use perceptron:
        else:
            # if user choose to use Pocket Algorithm
            if use_pocket:
                self.pocket(X, y)
            # if user do not use Pocket Algorithm, then just normal perceptron algorithm
            else:
                # We randomly initialize weights
                self.weight = np.random.rand(1, X.shape[1]+1)
                converged = False
                n = 0
                # Loop until convergence or limit of iteration times
                while converged == False:
                    if n == self.n_iter:
                        return
                    converged = True
                    for i in range(X.shape[0]):
                        # if the sample is not correct, update weights
                        if self.sign(y[i]) * self.sign(self._predict(X[i])) <= 0:
                            self.weight[0, 0] = self.weight[0, 0] + self.rate * self.sign(y[i])
                            self.weight[0, 1:] = self.weight[0, 1:] + self.rate * self.sign(y[i]) * X[i]
                            converged = False
                    n += 1

    # predict a single sample
    def _predict(self, x):
        return self.activation(np.dot(self.weight[0, 1:].T, x) + self.weight[0, 0])

    # predict dataset
    def predict(self, X):
        if self.use_mlp:
            self.forward(X)
            res = self.layers[-1].copy()
            #print(self.activation)
            if self.activator_name == 'sigmoid':
                for i in range(len(res)):
                    #print(res[i])
                    if res[i] >= 0.5:
                        res[i] = 1
                    else:
                        res[i] = 0
            return res
        else:
            res = []
            for i in range(X.shape[0]):
                res.append(self._predict(X[i]))
            return res

    # Transmit and calculate values of neurons from input layer till output layer
    def forward(self, inputs):
        self.layers[0] = inputs
        for i in range(len(self.weights)):
            next_layer = self.layers[i] @ self.weights[i] - self.weights0[i]
            self.layers[i+1] = self.activation(next_layer)

    # Back propagation algorithm
    # Input: y: label of sample previously forwarded
    def back_propagate(self, y):
        # calculate the error between label and estimated value
        err = y - self.layers[-1]
        # calculate gradient in output layer
        gradients = [(self.deactivation(self.layers[-1]) * err).sum(axis=0)]
        # Update 'b' in output layer
        self.weights0[-1] -= self.rate * gradients[-1]
        # Back-propagate from output layer to input layer, if no hidden layer, i.e. len(self.weights)-1=0, then this
        # step will not be applied, the algorithm will be just linear unit instead.
        for i in range(len(self.weights)-1, 0, -1):
            # calculate gradient of each layer
            gradient = np.sum(gradients[-1] @ self.weights[i].T * self.deactivation(self.layers[i]), axis=0)
            gradients.append(gradient)
            # update 'b's by layer
            self.weights0[i - 1] -= self.rate * gradients[-1] / 1.0
        gradients.reverse()
        # Update weights
        for i in range(len(self.weights)):
            x = np.mean(self.layers[i], axis=0)
            self.weights[i] += self.rate * x.reshape((-1, 1)) * gradients[i]



