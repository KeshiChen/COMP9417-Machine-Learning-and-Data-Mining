0. Environment:
    Python 3.6, Numpy 1.14.2, Pandas 0.23.0
1. Structure of the project:
    Algorithm implementation: MLPerceptron.py
    MPG (linear unit): mpg.py
    Linearly-separable Boolean function(with and without Pocket Algorithm): linearly_separable.py
    Non-linearly-separable Boolean function (using multilayer perceptron): non_linearly_separable.py
    Saved models are saved as ".pickle" files
    Testing results: result.txt
    Note: For Boolean functions, we use auto-generated datasets, so there will be no local data files for them.

3. Model files names:
MPG:
                linear_unit_mpg
Boolean function:
                no_pocket_noise0
                no_pocket_noise0.1
                no_pocket_noise0.3
                no_pocket_noise0.5
                pocket_noise0
                pocket_noise0.1
                pocket_noise0.3
                pocket_noise0.5
Non-linearly-separable:
                mlp_xor

1. How to run and test:
    Task 1
    Learn MPG using linear unit:
    1. Terminal command line description:
        python mpg.py
        -l[--load_model]: Whether to load existing model
        -c[--cross_val=] <k_value>: Whether to use cross-validation, and k value of k-fold, default 5(This may be very
                                   slow by its nature, depending on the k value you input)
        -t[--train_file=] <train_file>: File name of training dataset, default autoMpg.arff
        -p[--test_file=] <test_file>: File name of testing dataset, default autoMpg.arff (train/test splitted)
        -n[--iter_num=] <iterarion_times>: Training parameter, how many iterations, default 20000
        -r[--learn_rate=] <learning_rate>: Training parameter, the learning rate, default 0.01
    Note: We use random weight initialization, in order to avoid float length overflow, we logged the weight, but with
          limited testing times. So, if that problem still occurs, please run the program again.
    Task 2 & 3
    Learn 8-input linearly-separable Boolean function using perceptron training (with/without Pocket Algorithm):
    1. Terminal command line description:
        python linearly_separable.py
        -c[--compare]: Whether to see the performance comparison between algorithms using and not using Pocket Algorithm
                       under different noise rates
        Note: We train 10 times for each noise environment to collect average statistics, thus will be very slow.
        -l[--load_model=] <model_name>: Load existing model by name, see model names above.
        -n[--iter_num=] <iteration_times>: Training parameter, how many iterations, default 20000
        -r[--learn_rate=] <learning_rate>: Training parameter, the learning rate, default 0.01
        -o[--noise_rate=] <noise_rate>: User defined noise rate, default [0.0, 0.1, 0.3, 0.5].
        Note: If use user-defined noise rate, only one number will be used, overwriting the list by default.
              For example, if you input -o 0.6 then the noise_rates will be [0.6], instead of [0.0, 0.1, 0.3, 0.5].
        -b[--bool=] <'and'['or']>: Which Boolean function you want to train and test.
        Note: If -c and -l are empty, the program will simply test basic functionality of the implementation, with
              and without Pocket Algorithm, under each noise rates.

    Learn 8-input non-linearly-separable Boolean function using multilayer perceptron training algorithm:
    1. Terminal command line description:
        python non-linearly_separable.py
        -l[--load_model]: Whether to load existing model
        -n[--iter_num=] <iteration_times>: Training parameter, how many iterations, default 20000
        -r[--learn_rate=] <learning_rate>: Training parameter, the learning rate, default 0.01
        Note: If don't load existing model, the program will do the whole process from training to predicting, which
              will take around 10 minutes, please wait.

