from MLPerceptron import *
from tools import *
import sys, getopt
import numpy as np


def main(argv=None):
    helper = "linearly_separable.py -c[--compare] -l[--load_model] <model_name> -n[--iter_num] <iteration_times> -r[--learn_rate] \
    <learning_rate> -o[--noise_rate] <noise_rate> -b[--bool] <'and'['or']>"
    models = ["no_pocket_noise0.1", "no_pocket_noise0.3", "no_pocket_noise0.5", "no_pocket_noise0",
                                  "pocket_noise0.1", "pocket_noise0.3", "pocket_noise0.5", "pocket_noise0"]
    enable_model = False
    enable_compare = False
    model_filename = "no_pocket_noise0"
    n_iter = 20000
    eta = 0.01
    noise_rates = [0.0, 0.1, 0.3, 0.5]
    bfunc = 'and'
    if argv is None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], "hcl:n:r:o:b:", ["compare", "load_model=", "iter_num=", "learn_rate=",
                                                          "noise_rate=", "bool="])
    except getopt.GetoptError:
        print(helper)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helper)
            sys.exit()
        elif opt in ("-c", "--compare"):
            enable_compare = True
        elif opt in ("-l", "--load_model"):
            enable_model = True
            model_filename = str(arg)
        elif opt in ("-n", "--iter_num"):
            n_iter = int(arg)
        elif opt in ("-r", "--learn_rate"):
            eta = float(arg)
        elif opt in ("-o", "--noise_rate"):
            noise_rates = [float(arg)]
        elif opt in ("-b", "--bool"):
            bfunc = str(arg)
    if enable_model:
        if model_filename not in models:
            print("No such model, please select from the model list:")
            print(models)
            sys.exit()
        ands = generate_bool_dataset(256, 'and')
        dataset = np.array(ands)
        X = dataset[:, :-1]
        y = dataset[:, -1]
        pp = load_model(model_filename)
        y_pred = pp.predict(X)
        err = bool_func_error_calculator(y, y_pred)
        print("Error rate:", err)
    elif enable_compare:
        # Predict 8-input linear-separable boolean function with/without pocket algorithm
        if bfunc == 'and':
            data = generate_bool_dataset(256, 'and')
        elif bfunc == 'or':
            data = generate_bool_dataset(256, 'or')
        else:
            print("No such Boolean function, please select 'and' or 'or':")
            sys.exit()
        # Perceptron instance
        pp = MLPerceptron(n_iter=n_iter, rate=eta, activator='boolean', use_mlp=False)
        # Test and compare performance
        print("Training 10 times with noise rates in noise_rates list to compare performance with/without Pocket "
              "Algorithm, please wait.")
        for i in noise_rates:
            results = tester(pp, data, bfunc=bfunc, noise_rate=i, test_times=10)
    else:
        print("Simple functionality test with/without Pocket Algorithm, use user-defined noise rate.")
        if bfunc == 'and':
            data = generate_bool_dataset(256, 'and')
        elif bfunc == 'or':
            data = generate_bool_dataset(256, 'or')
        else:
            print("No such Boolean function, please select 'and' or 'or':")
            sys.exit()
        dataset = np.array(data)
        X = dataset[:, :-1]
        y = dataset[:, -1]
        for i in noise_rates:
            data_noised = np.array(noise_generator(data, bfunc, noise_rate=i))
            X_noised = data_noised[:, :-1]
            y_noised = data_noised[:, -1]
            pp1 = MLPerceptron(n_iter=n_iter, rate=eta, activator='boolean', use_mlp=False)
            pp2 = MLPerceptron(n_iter=n_iter, rate=eta, activator='boolean', use_mlp=False)
            print("Training perceptron without Pocket Algorithm, please wait...")
            t = time.clock()
            pp1.train(X_noised, y_noised, use_pocket=False)
            t_nopkt = time.clock() - t
            print("Training perceptron with Pocket Algorithm, please wait...")
            t = time.clock()
            pp2.train(X_noised, y_noised, use_pocket=True)
            t_pkt = time.clock() - t
            print("Testing...")
            y_pred1 = pp1.predict(X)
            y_pred2 = pp2.predict(X)
            err1 = bool_func_error_calculator(y, y_pred1)
            err2 = bool_func_error_calculator(y, y_pred2)
            print("Boolean function:", bfunc)
            print("Noise rate:", i)
            print("Without Pocket Algorithm:")
            print("Time cost:", t_nopkt)
            print("Error rate:", err1)
            print("With Pocket Algorithm:")
            print("Time cost:", t_pkt)
            print("Error rate:", err2)


if __name__ == "__main__":
    main()
