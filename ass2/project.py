import time
import pandas as pd
from scipy.io import arff
from sklearn.metrics import explained_variance_score
from tools import *
from MLPerceptron import *

# Generate random noise
# Input: X: dataset, bool_func: name of Boolean function ('and' for AND, 'or' for OR), noise_rate: noise rate (0 to 1.0)
def noise_generator(X, bool_func, noise_rate=0.1):
    copy = [list(i) for i in X]
    noise_num = int(noise_rate * len(copy))
    if bool_func == 'and':
        noises = [[0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 1]]
    elif bool_func == 'or':
        noises = [[0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0]]
    for i in range(noise_num):
        idx = random.randint(0, len(noises) - 1)
        noise = noises[idx]
        copy.append(noise)
    return copy


def label_race(row, col_name, value):
    if float(row[col_name]) == value:
        return 1
    else:
        return 0

# Get mean and standard deviation vectors from training set
def StandardScaler_Fit(X):
    mean = []
    std = []
    for i in range(X.shape[1]):
        mean.append(np.mean(X[:, i]))
        std.append(np.std(X[:, i]))
    return mean, std

# Transform original data into scaled data
def StandardScaler_Transform(X, mean, std):
    X_copy = X.copy()
    m = np.array(mean, dtype=np.float64)
    s = np.array(std, dtype=np.float64)
    #print(m, s)
    for i in range(X.shape[0]):
        X_copy[i,:] = (X_copy[i,:] - m) / s
    #print('X', X_copy)
    return X_copy

# Randomly split train and test sets, based on user input test size
def split_trian_test(x, y, test_size):
    test_num = int(x.shape[0] * test_size)
    test_idx = np.random.choice(x.shape[0], test_num).tolist()
    test_x = x[test_idx]
    test_y = y[test_idx]
    train_x = np.delete(x, test_idx, axis=0)
    train_y = np.delete(y, test_idx)
    return train_x, test_x, train_y, test_y


def bool_func_error_calculator(Y, y):
    length = Y.shape[0]
    error = 0.0
    for i in range(length):
        #print(Y[i], y[i])
        sign = 1 if y[i] >= 0.5 else 0
        if Y[i] != sign:
            error+=1.0
    return error/float(length)

# Test performance of perceptron training algorithm with/without Pocket Algorithm
# Input: pp: perceptron instance, data: test dataset, bfunc: Boolean function name ('and' for AND, 'or' for OR),
#            noise_rate: noise rate (0.0 to 1.0), test_times: integer, how many times will be run for each algorithm
# Dataset: Boolean function
# Environment: User input different level of noise
# Test time: User input times for each algorithm
# Measure: Average time cost, average error rate
def tester(pp, data, bfunc, noise_rate=0.1, test_times = 1):
    dataset = np.array(data)
    data_noised = np.array(noise_generator(data, bfunc, noise_rate=noise_rate))
    X = dataset[:, :-1]
    y = dataset[:, -1]
    X_noised = data_noised[:, :-1]
    y_noised = data_noised[:, -1]
    avg_time_pocket = 0.0
    avg_time_no_pocket = 0.0
    avg_err_rate_pocket = 0.0
    avg_err_rate_nopkt = 0.0
    for i in range(test_times):
        t0 = time.clock()
        pp.train(X_noised, y_noised, use_pocket=True)
        t1 = time.clock()
        save_model("pocket_noise"+str(noise_rate), pp)
        avg_time_pocket += t1 - t0
        pred_pkt = pp.predict(X)
        t0 = time.clock()
        pp.train(X_noised, y_noised, use_pocket= False)
        t1 = time.clock()
        save_model("no_pocket_noise" + str(noise_rate), pp)
        avg_time_no_pocket += t1 - t0
        pred_nopkt = pp.predict(X)
        avg_err_rate_nopkt += bool_func_error_calculator(y, pred_nopkt)
        avg_err_rate_pocket += bool_func_error_calculator(y, pred_pkt)
    avg_time_pocket = avg_time_pocket / test_times
    avg_time_no_pocket = avg_time_no_pocket / test_times
    avg_err_rate_nopkt = avg_err_rate_nopkt / test_times
    avg_err_rate_pocket = avg_err_rate_pocket / test_times
    print("Boolean function:", bfunc)
    print("Noise rate:", noise_rate)
    print("Test times:", test_times)
    print("Average time cost without pocket algorithm:", avg_time_no_pocket)
    print("Average time cost with pocket algorithm:", avg_time_pocket)
    print("Average error rate without pocket algorithm:", avg_err_rate_nopkt)
    print("Average error rate with pocket algorithm:", avg_err_rate_pocket)
    output_file = "result.txt"
    print("Boolean function:", bfunc, file=open(output_file, "a+"))
    print("Noise rate:", noise_rate, file=open(output_file, "a+"))
    print("Test times:", test_times, file=open(output_file, "a+"))
    print("Average time cost without pocket algorithm:", avg_time_no_pocket, file=open(output_file, "a+"))
    print("Average time cost with pocket algorithm:", avg_time_pocket, file=open(output_file, "a+"))
    print("Average error rate without pocket algorithm:", avg_err_rate_nopkt, file=open(output_file, "a+"))
    print("Average error rate with pocket algorithm:", avg_err_rate_pocket, file=open(output_file, "a+"))
    print("", file=open(output_file, "a+"))
    return avg_time_no_pocket, avg_time_pocket, avg_err_rate_nopkt, avg_err_rate_pocket


def MSE(target, predictions):
    squared_deviation = np.power(target - predictions[:, 0], 2)
    return np.mean(squared_deviation)


filename = "autoMpg.arff"
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
test_size = 0.2
random_state = 3
x_train, x_test, y_train, y_test = split_trian_test(x, y, test_size=test_size)
mean, std = StandardScaler_Fit(x_train)
X_train_std = StandardScaler_Transform(x_train, mean, std)
X_train_std = np.array(X_train_std, dtype=np.float64)

X_test_std = StandardScaler_Transform(x_test, mean, std)
X_test_std = np.array(X_test_std, dtype=np.float64)
# perceptron

n_iter = 20000
eta = 0.01
random_state = 0
# Predict MPG using Multilayer Perceptron
pp = MLPerceptron((25, 1), n_iter=n_iter, rate=eta, activator='linear')
t = time.clock()
pp.train(X_train_std, y_train)
t = time.clock() - t
save_model("linear_unit_mpg", pp)
print('Linear unit on MPG dataset:', file=open("result.txt", "a+"))
print("time:",t)
print("time:",t, file=open("result.txt", "a+"))
y_predicted = pp.predict(X_test_std)
y_test = np.asarray(y_test, dtype=np.float)
print(y_predicted)
score = explained_variance_score(y_true=y_test, y_pred=y_predicted)
print('explained_variance_score', score)
print('explained_variance_score', score, file=open("result.txt", "a+"))
for i in range(y_test.shape[0]):
    print(y_test[i], y_predicted[i])
score = MSE(y_test, y_predicted)
print('MSE score:', score)
print('MSE score:', score, file=open("result.txt", "a+"))
print("", file=open("result.txt", "a+"))

noise_rate = 0.1
# Predict 8-input linear-separable boolean function using pocket algorithm
ands = generate_bool_dataset(256, 'and')
ors = generate_bool_dataset(256, 'or')
and_dataset = np.array(ands)
# Generate random noises
and_dataset_noised = np.array(noise_generator(ands, 'and', noise_rate=noise_rate))
or_dataset = np.array(ors)
# Generate random noises
or_dataset_noised = np.array(noise_generator(ors, 'or', noise_rate=noise_rate))
X_and = and_dataset[:, :-1]
y_and = and_dataset[:, -1]
X_and_noised = and_dataset_noised[:, :-1]
y_and_noised = and_dataset_noised[:, -1]
X_or = or_dataset[:, :-1]
y_or = or_dataset[:, -1]
X_or_noised = or_dataset_noised[:, :-1]
y_or_noised = or_dataset_noised[:, -1]
pp = MLPerceptron(n_iter=n_iter, rate=eta, activator='boolean', use_mlp=False)
# Test and compare performance
noise_rates = [0.0, 0.1, 0.3, 0.5]
for i in noise_rates:
    results = tester(pp, ands, bfunc='and', noise_rate=i, test_times=10)

# Predict 8-input non-linear-separable boolean function using Multilayer Perceptron
xors = generate_bool_dataset(256, 'xor')
xor_train = list(xors)
for i in range(50):
    xor_train.append([1, 1, 1, 1, 1, 1, 1, 1, 0])
    xor_train.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
xor_dataset = np.array(xors)
xor_trainset = np.array(xor_train)
X_xor = xor_dataset[:, :-1]
X_xor_trian = xor_trainset[:, :-1]
y_xor = xor_dataset[:, -1]
y_xor_train = xor_trainset[:, -1]
n = MLPerceptron((8, 20, 1), activator='sigmoid', n_iter=20000, rate=0.01)
t = time.clock()
n.train(X_xor_trian, y_xor_train)
t = time.clock() - t
save_model("mlp_xor", n)
print("Multilayer Perceptron to learn XOR:", file=open("result.txt", "a+"))
print("Time cost:", t)
print("Time cost:", t, file=open("result.txt", "a+"))
xor_test_x = np.array(X_xor)
xor_test_y = np.array(y_xor)
pred = n.predict(xor_test_x)
print("Error rate:",  bool_func_error_calculator(xor_test_y, pred))
print("Error rate:", bool_func_error_calculator(xor_test_y, pred), file=open("result.txt", "a+"))
print("", file=open("result.txt", "a+"))
