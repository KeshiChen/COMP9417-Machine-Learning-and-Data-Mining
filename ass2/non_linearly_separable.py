from tools import *
from MLPerceptron import *
import sys, getopt


def main(argv=None):
    enable_model = False
    n_iter = 20000
    eta = 0.01
    bfunc = 'xor'
    helper = "non_linearly_separable.py -l[--load_model] -b[--bfunc=] <xor[and][or]> -n[--iter_num=] <iteration_times>"\
             " -r[--learn_rate=] <learning_rate> "
    if argv is None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], "hlb:n:r:", ["load_model", "iter_num=", "learn_rate=",
                                                          "noise_rate=", "bool="])
    except getopt.GetoptError:
        print(helper)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helper)
            sys.exit()
        elif opt in ("-l", "--load_model"):
            enable_model = True
        elif opt in ("-b", "--bfunc"):
            bfunc = str(arg)
        elif opt in ("-n", "--iter_num"):
            n_iter = int(arg)
        elif opt in ("-r", "--learn_rate"):
            eta = float(arg)
    if enable_model:
        data = generate_bool_dataset(256, 'xor')
        dataset = np.array(data)
        X_test = dataset[:, :-1]
        y = dataset[:, -1]
        pp = load_model("mlp_xor")
        y_pred = pp.predict(X_test)
        print("Learn 8-input XOR function using multilayer perceptron:")
        print("Layers (input, hidden, output):", (8, 6, 1))
        print("Iterations:", 20000)
        print("Learning rate:", 0.01)
        print("Error rate:", bool_func_error_calculator(y, y_pred))
    else:
        #  8-input non-linear-separable boolean function using Multilayer Perceptron
        if bfunc == "xor":
            data = generate_bool_dataset(256, 'xor')
            data_train = list(data)
            for i in range(50):
                data_train.append([1, 1, 1, 1, 1, 1, 1, 1, 0])
                data_train.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
            dataset = np.array(data)
            data_noised = np.array(noise_generator(data, bfunc, noise_rate=0.5))
            X_train_noised = data_noised[:, :-1]
            y_train_noised = data_noised[:, -1]
            trainset = np.array(data_train)
            X_test = dataset[:, :-1]
            X_train = trainset[:, :-1]
            y = dataset[:, -1]
            y_train = trainset[:, -1]
        elif bfunc == "and":
            data = generate_bool_dataset(256, 'and')
            dataset = np.array(data)
            X_train = X_test = dataset[:, :-1]
            y_train = y = dataset[:, -1]
        elif bfunc == "or":
            data = generate_bool_dataset(256, 'or')
            dataset = np.array(data)
            X_train = X_test = dataset[:, :-1]
            y_train = y = dataset[:, -1]
        else:
            print("No such Boolean function, please select 'and', 'or' or 'xor':")
            sys.exit()
        n = MLPerceptron((8, 6, 1), activator='sigmoid', n_iter=n_iter, rate=eta)
        print("Learn 8-input " + bfunc + " function using multilayer perceptron:")
        print("Layers (input, hidden, output):", (8, 6, 1))
        print("Iterations:", n_iter)
        print("Learning rate:", eta)
        print("Training...")
        t = time.clock()
        # n.train(X_train, y_train)
        n.train(X_train_noised, y_train_noised)
        t = time.clock() - t
        print("Training complete.")
        # save_model("mlp_xor", n)
        # print("Multilayer Perceptron to learn XOR:", file=open("result.txt", "a+"))
        print("Time cost:", t)
        # print("Time cost:", t, file=open("result.txt", "a+"))
        # xor_test_x = np.array(X_test)
        # xor_test_y = np.array(y)
        pred = n.predict(X_test)
        #print(pred)
        print("Error rate:",  bool_func_error_calculator(y, pred))
        # print("Error rate:", bool_func_error_calculator(xor_test_y, pred), file=open("result.txt", "a+"))
        # print("", file=open("result.txt", "a+"))


if __name__ == "__main__":
    main()