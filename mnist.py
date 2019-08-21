""" Imports """
from timeit import default_timer as timer
import pandas as pd
from pprint import pprint
import itertools
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import tensorflow as tf
import tensorflow_datasets as tfds  # getting MNIST from there
"""
Goal: Find a good model to learn a well-known Mnist dataset of hand-written numbers
Details: try many models with many different hyperparameters to be able to find the best model
Result: Found a model with Train and Validate accuracy of > 99.99% and Test accuracy ~ 99.85%

Each image is 28x28 pixels, or 784 pixels
Each pixel is greyscale - 0 (black) to 255 (white)

Dataset can be found usually on Windows in C:/Users/<user>/tensorflow_datasets

Number of inputs: 784 (flattened image)
Number of outputs: 10 (0-9)
Encoding for outputs - one-hot encoding, 0 - [1,0,0,0,0,0,0,0,0,0] etc.
Activation function last layer: softmax
"""

""" Settings / Hyperparameters - see explanation in README.md """
FRACTION_VALIDATE = 0.1

# Maximum number of epochs.  Currently by having a very high value, practically not used and relying on EarlyStopping.
# See reasons in README.md
MAX_NUM_EPOCHS = 1000

validate_loss_improve_deltas = [0.0001, 0.00001]

# validate_loss_improve_patiences = [7, 10, 15]
validate_loss_improve_patiences = [10, 15]

improve_restore_best_weights_values = [True]

batch_sizes = [200, 400, 600]

hidden_widths = [200, 450, 784]

nums_layers = [4, 5]

functions = ['sigmoid', 'tanh', 'relu', 'softmax']
functions = ['sigmoid', 'tanh', 'relu']

learning_rates = [0.001, 0.0005, 0.00001]  # default in tf.keras.optimizers.Adam is 0.001
learning_rates = [0.001, 0.0005]  # default in tf.keras.optimizers.Adam is 0.001


def acquire_data():
    mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

    print(f'Name: {mnist_info.full_name}')
    print(f'Description: {mnist_info.description}')
    print(f'Size in bytes: {mnist_info.size_in_bytes}')
    print(f'Features:')
    pprint(mnist_info.features)

    print(f"Num train and validation samples: {mnist_info.splits['train'].num_examples}, "
          f"Num training samples: {mnist_info.splits['test'].num_examples}")

    return mnist_dataset, mnist_info.splits['train'].num_examples, mnist_info.splits['test'].num_examples


def scale_0_to_1(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.  # colors are 0 - 255, we are scaling it to 0 to 1 range
    return image, label


def scale_data(mnist_train_and_valid_orig, mninst_test_orig):
    return mnist_train_and_valid_orig.map(scale_0_to_1()), \
           mninst_test_orig.map(scale_0_to_1())


def split_data(train_data, num_train_valid_examples):
    num_validation_samples = tf.cast(FRACTION_VALIDATE * num_train_valid_examples, tf.int64)
    print(f"Num validation samples: {num_validation_samples}, "
          f"Num training samples: {num_train_valid_examples - num_validation_samples}")

    # returns validation and training datasets
    return train_data.take(num_validation_samples), train_data.skip(num_validation_samples)


def prepare_data(mnist_dataset, num_train_valid_examples, num_test_examples, batch_size, shuffle_seed):
    train_valid_data = mnist_dataset['train']
    test_data = mnist_dataset['test']

    # Scale data
    train_valid_data = train_valid_data.map(scale_0_to_1)
    test_data = test_data.map(scale_0_to_1)

    # Shuffle training data
    train_valid_data = train_valid_data.shuffle(num_train_valid_examples, seed=shuffle_seed)
    test_data = test_data.shuffle(num_test_examples, seed=shuffle_seed)

    # Split training and validation data
    valid_data, train_data = split_data(train_valid_data, num_train_valid_examples)

    # Batch train data
    train_data = train_data.batch(batch_size)
    valid_data = valid_data.batch(num_train_valid_examples)  # giving a too big of a number = keep validate in one batch
    test_data = test_data.batch(num_test_examples)  # keep test in one batch

    # next will load next batch, since only one, will load everything
    valid_inputs, valid_targets = next(iter(valid_data))

    return train_data, valid_inputs, valid_targets, test_data


def prepare_model(in_dic):
    output_size = 10

    # Add input layer
    model = tf.keras.Sequential([
                                tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # becomes size 784
                                ])

    # add hidden layers
    for i in range(in_dic['Num layers'] - 2):
        model.add(tf.keras.layers.Dense(in_dic['Hidden width'], activation=in_dic['Hidden funcs'][i]))

    # Add output layer
    model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

    """ Compile the model
    Adam - optimizer that uses both momentum and learning rate schedule - combines best from both worlds
    Loss sparse_categorical_crossentropy - for classification problems that were already one-hoted
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(
                                                    learning_rate=in_dic['Learning rate']),
                                                    loss='sparse_categorical_crossentropy',
                                                    metrics=['accuracy'])
    return model


def single_model(train_data, valid_inputs, valid_targets, test_data, in_dic):
    """
    Perform a run of a single model, returns dictionary with results
    """
    out_dic = {}

    model = prepare_model(in_dic)

    # Stops the model when val_loss doesn't improve by min_delta for patience loops.
    #  Possibly restores weights to best value of val_loss
    early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=in_dic['Validate loss improvement delta'],
                                                      patience=in_dic['Validate loss improvement patience'],
                                                      restore_best_weights=in_dic['Restore best weights'])
    start = timer()
    history = model.fit(train_data,
                        epochs=in_dic['Max num epochs'],
                        callbacks=[early_callback],
                        validation_data=(valid_inputs, valid_targets),
                        verbose=2)
    end = timer()

    # Test the model
    test_loss, test_accuracy = model.evaluate(test_data)

    actual_num_epochs = len(history.history["val_accuracy"])

    # Prepare output of results - see explanation of outputs in README.md
    out_dic['Test Accuracy'] = round(test_accuracy, 4)
    out_dic['Test Loss'] = round(test_loss, 4)
    out_dic['Train Time'] = round(end - start, 3)
    out_dic['Loss * Time'] = round(out_dic['Test Loss'] * (end - start), 4)
    out_dic['Validate Accuracy'] = history.history["val_accuracy"][-1].round(4)
    out_dic['Train Accuracy'] = history.history["accuracy"][-1].round(4)
    out_dic['Validate Loss'] = history.history["val_loss"][-1].round(4)
    out_dic['Train Loss'] = history.history["loss"][-1].round(4)
    out_dic['Num epochs'] = actual_num_epochs
    out_dic['Average epoch time'] = round((end - start) / actual_num_epochs, 4)

    return out_dic


def do_numerous_loops(num_loops=1, given_dic=None):
    """ Performs numerous models with different inputs / hyperparameters.
        Outputs to output files - see README.md for explanation """

    results = []
    # Inputs are given in 2 ways:
    #  1. Mostly for local testing - explicit by giving given_dic, and only 1 run is done (possibly with numerous loops)
    #  2. By not giving given_dic, and then numerous inputs / hyperparameters are taken from the settings
    if given_dic:
        in_dic = given_dic
        local_validate_loss_improve_deltas = [in_dic['Validate loss improvement delta']]
        local_validate_loss_improve_patiences = [in_dic['Validate loss improvement patience']]
        local_improve_restore_best_weights_values = [in_dic['Restore best weights']]
        local_batch_sizes = [in_dic['Batch size']]
        local_hidden_widths = [in_dic['Hidden width']]
        local_nums_layers = [in_dic['Num layers']]
        local_functions = [in_dic['Hidden funcs']]
        local_learning_rates = [in_dic['Learning rate']]
    else:
        # MAX_NUM_EPOCHS is constant - no loops on different values
        in_dic = {'Max num epochs': MAX_NUM_EPOCHS,
                  'Shuffle seed': 100}
        local_validate_loss_improve_deltas = validate_loss_improve_deltas
        local_validate_loss_improve_patiences = validate_loss_improve_patiences
        local_improve_restore_best_weights_values = improve_restore_best_weights_values
        local_batch_sizes = batch_sizes
        local_hidden_widths = hidden_widths
        local_nums_layers = nums_layers
        local_functions = functions
        local_learning_rates = learning_rates

    # For printing purposes, calculate number of models to be done
    #  (not including activation functions that are harder to calculate and also depend on number of layers)
    num_regressions_without_functions = num_loops * \
                                        len(local_validate_loss_improve_deltas) * \
                                        len(local_validate_loss_improve_patiences) * \
                                        len(local_improve_restore_best_weights_values) * \
                                        len(local_batch_sizes) * \
                                        len(local_hidden_widths) * \
                                        len(local_nums_layers) * \
                                        len(local_learning_rates)
    print(f'To perform number regressions: {num_regressions_without_functions} '
          f'with layers: {local_nums_layers} and functions: {local_functions}')

    # acquiring data is done once to save time
    mnist_data, num_train_valid_examples, num_test_examples = acquire_data()

    # initiate best values that will be overridden when finding a good result
    best_test_accuracy = {'Test Accuracy': 0.001}
    best_test_loss = {'Test Loss': 100000}
    best_loss_efficiency = {'Loss * Time': 100000}

    num_model_trainings = 0
    time_run_started = timer()

    for batch_size in local_batch_sizes:
        in_dic['Batch size'] = batch_size
        if num_loops == 1:
            # to save time not to do it every time if there is only 1 loop
            in_dic['Shuffle seed'] = 1
            # Shuffling is affected by 2 hyperparameters: batch_size (therefore needs to be done per batch size), and
            #   Shuffle seed that changes per loop. Therefore needs to be done later in the loops pass,
            #   however, if there is only 1 loop, shuffling can be done once here.
            train_data, valid_inputs, valid_targets, test_data = prepare_data(mnist_data,
                                                                              num_train_valid_examples,
                                                                              num_test_examples,
                                                                              batch_size,
                                                                              in_dic['Shuffle seed'])

        for validate_loss_improve_delta in local_validate_loss_improve_deltas:
            in_dic['Validate loss improvement delta'] = validate_loss_improve_delta
            for validate_loss_improve_patience in local_validate_loss_improve_patiences:
                in_dic['Validate loss improvement patience'] = validate_loss_improve_patience
                for improve_restore_best_weights in local_improve_restore_best_weights_values:
                    in_dic['Restore best weights'] = improve_restore_best_weights
                    for num_layers in local_nums_layers:
                        in_dic['Num layers'] = num_layers
                        # if given specific run, don't do a product of all activation functions, otherwise perform
                        #   a product of all activation functions to put each one of the functions in each hidden layer
                        if not given_dic:
                            funcs_product = itertools.product(local_functions, repeat=(num_layers - 2))
                        else:
                            funcs_product = [in_dic['Hidden funcs']]
                        print(f'funcs_product: {funcs_product}')
                        for hidden_funcs in funcs_product:
                            in_dic['Hidden funcs'] = hidden_funcs
                            for hidden_width in local_hidden_widths:
                                in_dic['Hidden width'] = hidden_width
                                for learning_rate in local_learning_rates:
                                    in_dic['Learning rate'] = learning_rate
                                    num_model_trainings += 1
                                    # Accumulate values are used for multiple loops with same hyperparameters
                                    #  (besides seed that's per loop)
                                    accum_test_accuracy = 0
                                    accum_test_loss = 0
                                    accum_loss_efficiency = 0

                                    #initiate values to be used initiated inside the loops and used outside
                                    # to quite a warning
                                    result = {}
                                    train_data = 0
                                    valid_inputs = 0
                                    valid_targets = 0
                                    test_data = 0

                                    for loop in range(1, num_loops+1):
                                        if num_loops > 1:
                                            # if more than 1 loops, to allow for different seed, prepare data per loop
                                            in_dic['Shuffle seed'] = loop
                                            train_data, valid_inputs, valid_targets, test_data = \
                                                prepare_data(mnist_data,
                                                             num_train_valid_examples,
                                                             num_test_examples,
                                                             batch_size,
                                                             in_dic['Shuffle seed'])

                                        # Printing progess of epochs, models, time and loop
                                        time_running_sec = timer() - time_run_started
                                        print(f'Model {num_model_trainings}, loop {loop}/{num_loops}, '
                                              f'total time min: {round(time_running_sec / 60, 1)}, '
                                              f'total time hours: {round(time_running_sec / 60 / 60, 2)}: '
                                              f'seconds per model: {round(time_running_sec / num_model_trainings)} '
                                              f'====================================')
                                        out_dic = single_model(train_data,
                                                               valid_inputs,
                                                               valid_targets,
                                                               test_data,
                                                               in_dic)
                                        # result for purposes of output to file contains both inputs and outputs
                                        result = in_dic.copy()
                                        result.update(out_dic)
                                        results.append(result)
                                        print(f'\nCURRENT: {result}')

                                        # calculate total for all loops of same model
                                        accum_test_accuracy += result['Test Accuracy']
                                        accum_test_loss += result['Test Loss']
                                        accum_loss_efficiency += result['Loss * Time']

                                    # After finishing all loops for a given model,
                                    #  update and print values of averages for loops - relevant only to
                                    #  printing and outputting to file of best values.  Full contains all specific
                                    #  runs of the loop
                                    if (accum_test_accuracy / num_loops) > best_test_accuracy['Test Accuracy']:
                                        best_test_accuracy = \
                                            {'Average loop result': round(accum_test_accuracy / num_loops, 4)}
                                        best_test_accuracy.update(result)
                                    if (accum_test_loss / num_loops) < best_test_loss['Test Loss']:
                                        best_test_loss = \
                                            {'Average loop result': round(accum_test_loss / num_loops, 4)}
                                        best_test_loss.update(result)
                                    if (accum_loss_efficiency / num_loops) < best_loss_efficiency['Loss * Time']:
                                        best_loss_efficiency = \
                                            {'Average loop result': round(accum_loss_efficiency / num_loops, 4)}
                                        best_loss_efficiency.update(result)

                                    print("Finished all loops +++++++++++++++++++++++++++++++++++++++++++++")
                                    print(f'BEST TEST ACCURACY:     {best_test_accuracy}')
                                    print(f'BEST TEST LOSS:         {best_test_loss}')
                                    print(f'BEST LOSS EFFICIENCY:   {best_loss_efficiency}')

    # Output all results to full.xlsx
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'Total number of models trained: {num_model_trainings} with {num_loops} loops per model')
    pf = pd.DataFrame(results)
    print(f'ALL RESULTS:')
    print(pf.to_string())
    pf.to_excel("output\\full.xlsx")

    time_running_sec = timer() - time_run_started

    # Output all inputs / hyperparameters to hyperparams.xlsx
    hyperparams = {
        'Num Model Trainings': num_model_trainings,
        'Num Loops per model': num_loops,
        'Total time minutes': round(time_running_sec / 60, 1),
        'Total time hours': round(time_running_sec / 60 / 60, 2),
        'Seconds per model': round(time_running_sec / num_model_trainings),
        'Max Num Epochs': in_dic['Max num epochs'],
        'Batch sizes': local_batch_sizes,
        'Hidden Widths': local_hidden_widths,
        'Nums layers': local_nums_layers,
        'Functions': local_functions,
        'Learning rates': local_learning_rates,
        'Improvement deltas': local_validate_loss_improve_deltas,
        'Improvement patience': local_validate_loss_improve_patiences,
        'Improvement restore weights': local_improve_restore_best_weights_values}

    pf = pd.DataFrame([hyperparams])
    print(f'HYPERPARAMS:')
    print(pf.to_string())
    pf.to_excel("output\\hyperparams.xlsx")

    # Output all best results (in 3 categories, see explanation in README.md) to best.xlsx
    best_test_accuracy_with_type = {'Type': 'TEST ACCURACY'}
    best_test_accuracy_with_type.update(best_test_accuracy)
    best_test_loss_with_type = {'Type': 'TEST LOSS'}
    best_test_loss_with_type.update(best_test_loss)
    best_loss_efficiency_with_type = {'Type': 'TEST LOSS EFFICIENCY'}
    best_loss_efficiency_with_type.update(best_loss_efficiency)

    pf = pd.DataFrame([best_test_accuracy_with_type,
                       best_test_loss_with_type,
                       best_loss_efficiency_with_type])
    print(f'BEST RESULTS:')
    print(pf.to_string())
    pf.to_excel("output\\best.xlsx")


""" Ways to run:
1. Regular way - numerous models for later comparison between them.
    - Update settings / hyperparameters options from beginning of file.  Each combination will be done
    - Run do_numerous_loops with given number of loops per model (usually 1 since very time consuming even with 1 loop)
        and without a dictionary
2. Debug / confirmation / researching one model way:
    - Give number of loops (often > 1 to confirm behavior / results)
    - Give explicit dictionary of model to run
    Values given below are for some of the best models
"""
# do_numerous_loops(1)

# """
# Best 4 layers
do_numerous_loops(3, {'Validate loss improvement delta': 0.0001,
                      'Validate loss improvement patience': 10,
                      'Restore best weights': True,
                      'Max num epochs': 1000,
                      'Batch size': 450,  # 200
                      'Num layers': 4,
                      'Hidden funcs': ('relu', 'relu'),
                      'Hidden width': 450,
                      'Learning rate': 0.001})
# """
"""
# 2nd best 4 layers
do_numerous_loops(3, {'Validate loss improvement delta': 0.0001,
                      'Validate loss improvement patience': 10,
                      'Restore best weights': True,
                      'Max num epochs': 1000,
                      'Batch size': 200,
                      'Num layers': 4,
                      'Hidden funcs': ('tanh', 'relu'),
                      'Hidden width': 200,
                      'Learning rate': 0.001})
"""
"""
# Best 5 layers
do_numerous_loops(3, {'Validate loss improvement delta': 0.0001,
                      'Validate loss improvement patience': 10,
                      'Restore best weights': True,
                      'Max num epochs': 1000,
                      'Batch size': 450,
                      'Num layers': 5,
                      'Hidden funcs': ('tanh', 'relu', 'tanh'),
                      'Hidden width': 450,
                      'Learning rate': 0.001})
"""
"""
do_numerous_loops(3, {'Validate loss improvement delta': 0.0001,
                      'Validate loss improvement patience': 10,
                      'Restore best weights': True,
                      'Max num epochs': 1000,
                      'Batch size': 300,
                      'Num layers': 5,
                      'Hidden funcs': ('relu', 'tanh', 'sigmoid'),
                      'Hidden width': 300,
                      'Learning rate': 0.001})
"""