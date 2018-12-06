from MyPerceptron import *
from sklearn.model_selection import train_test_split
from sklearn import datasets

n_iter = 100000
eta = 0.01
test_size=0.2
random_state=3

digits = datasets.load_digits()
#print(digits.data)
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=random_state)
#print(X_train[0,:])
mp = MyPerceptron(n_iter= n_iter, eta= eta, converge_threshold= 0.00001)
mp.train(X_train, y_train)
predicted = mp.predict(X_test)
for i in range(y_test.shape[0]):
    print(y_test[i], predicted[i])