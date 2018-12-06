from MLPerceptron import *
from tools import *
import sys,getopt
import numpy as np


def main(argv=None):
    helper = "mpg.py -l[--load_model] -c[--cross_val=] <k_value> -t[--train_file=] <train_file>" \
             " -p[--test_file=] <test_file> -n[--iter_num=] <iterarion_times> -r[--learn_rate=] <learning_rate>"
    enable_cross_validation = False
    enable_model = False
    test_filename = ""
    train_filename = "autoMpg.arff"
    k = 5
    n_iter = 20000
    eta = 0.01
    if argv is None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], "hlc:t:p:n:r:", ["load_model", "cross_val=", "train_file=", "test_file=",
                                                          "iter_num=", "learn_rate="])
    except getopt.GetoptError:
        print(helper)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helper)
            sys.exit()
        elif opt in ("-l", "--load_model"):
            enable_model = True
        elif opt in ("-c", "--cross_val"):
            enable_cross_validation = True
            k = int(arg)
        elif opt in ("-t", "--train_file"):
            train_filename = arg
        elif opt in ("-p", "--test_file"):
            test_filename = arg
        elif opt in ("-n", "--iter_num"):
            n_iter = int(arg)
        elif opt in ("-r", "--learn_rate"):
            eta = float(arg)

    # Learn MPG using linear unit
    if enable_model:
        pp1 = load_model("linear_unit_mpg")
        if len(test_filename) > 0:
            x_test, y_test = preprocessing(test_filename)
        else:
            x, y = preprocessing(train_filename)
            x_train, x_test, y_train, y_test = split_train_test(x, y, test_size=0.2)
        y_pred = pp1.predict(x_test)
        print("Explained variance score:", exp_var(y_test, y_pred))
    elif enable_cross_validation:
        n = k
        pp = MLPerceptron((25, 1), n_iter=n_iter, rate=eta, activator='linear')
        x, y = preprocessing(train_filename)
        t = time.clock()
        score = cross_validation(k, n, x, y, pp)
        t = time.clock() - t
        print("time:", t)
        # print('Linear unit on MPG dataset:', file=open("result.txt", "a+"))
        print("Cross-validation score:", score)
        # print("Cross-validation score:", score, file=open("result.txt", "a+"))
        # print("", file=open("result.txt", "a+"))
        # save_model("linear_unit_mpg", pp)
    else:
        pp = MLPerceptron((25, 1), n_iter=n_iter, rate=eta, activator='linear')
        if len(test_filename) > 0:
            x_train, y_train = preprocessing(train_filename)
            x_test, y_test = preprocessing(test_filename)
        else:
            x, y = preprocessing(train_filename)
            x_train, x_test, y_train, y_test = split_train_test(x, y, test_size=0.2)
        print("Whole process from training to predicting:")
        pp.train(x_train, y_train)
        y_pred = pp.predict(x_test)
        print("Explained variance score:", exp_var(y_test, y_pred))
    sys.exit()


if __name__ == "__main__":
    main()
