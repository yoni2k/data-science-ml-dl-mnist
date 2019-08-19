"""
Each image is 28x28 pixels, or 784 pixels
Each pixel is greyscale - 0 (black) to 255 (white)

Dataset can be found usually on Windows in C:/Users/<user>/tensorflow_datasets

Number of inputs: 784 (flattened image)
Number of outputs: 10 (0-9)
Encoding for outputs - one-hot encoding, 0 - [1,0,0,0,0,0,0,0,0,0] etc.
Number of hidden layers: 2
Activation function last layer: softmax

TODOs:
- After reading solutions:
  - try with much larger hidden width - supposed to give even better results, but much slower
- Make sure the function works also when ran regularly and not with given dic
- Write comments
- Try extreme values to see what effect they have
- TODOs
- Write general README
- Commit also Jupyter notebook with solution

Questions to ask:
- How come even when seeding getting different results?
- Different functions, when does each one make sense?
- Test - how come gives different results every time
- How to do it in the cloud - AWS AMI images don't have 2.0, containers also?
"""

""" Imports """
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import tensorflow as tf
import tensorflow_datasets as tfds  # getting MNIST from there
from timeit import default_timer as timer
import pandas as pd
from pprint import pprint
import itertools

""" Settings """
FRACTION_VALIDATE = 0.1

# Current conclusions - with Delta 0.001 and Patience 3, need 25 and possibly 30
# Later conclusion - depends very much on so many other parameters (sometimes too much, sometimes too low),
#   so decided to basically leave unlimited and rely on StopFunction only
MAX_NUM_EPOCHS = 1000

accuracy_improvement_deltas = [0.0001]
# accuracy_improvement_deltas = [0.001, 0.0001]

accuracy_improvement_patiences = [3]
# accuracy_improvement_patiences = [3, 4]

# Tried before [100, 1000], [500], [50, 100, 250, 500], [50, 75, 100, 170, 250], [100, 150, 200], [400], [300]
# Current conclusion - 100 seems the best, but 50-250 all give good results, 200 seems as good as lower. Possibly higher also OK, but takes more time when climing up
# Conclusion later - 400 seems slightly faster and slightly more accurate than 200, so leaving with 300 for now
batch_sizes = [300]
# batch_sizes = [200, 300, 400]

## Tried before [10, 64], [50], [25, 50, 75], [500], [200]
# Current conclusion - 75 gives the best results from [25, 50, 75], need to try heigher also
# Later conclusion: 500 seems too much - worse results and slower
#hidden_widths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
#hidden_widths = [100, 200, 300]
hidden_widths = [200]

# Tried [3, 4, 5, 6], [5], [4]
# Current conclusion - 4 layers seems enough, although with 5 layers get similar results (and sometimes takes longer)
#nums_layers = [2, 3, 4, 5, 6, 10]
nums_layers = [4]

functions = ['sigmoid', 'tanh', 'relu', 'softmax']
#functions = ['relu', 'sigmoid', 'tanh']
#functions = ['relu']

# learning_rates = [0.0005, 0.001, 0.005]  # default in tf.keras.optimizers.Adam is 0.001
learning_rates = [0.001]  # default in tf.keras.optimizers.Adam is 0.001


def acquire_data():
    mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

    print(f'Name: {mnist_info.full_name}')
    print(f'Description: {mnist_info.description}')
    print(f'Size in bytes: {mnist_info.size_in_bytes}')
    print(f'Features:')
    pprint(mnist_info.features)

    print(f"Num train and validation samples: {mnist_info.splits['train'].num_examples}, Num training samples: {mnist_info.splits['test'].num_examples}")

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
    print(f"Num validation samples: {num_validation_samples}, Num training samples: {num_train_valid_examples - num_validation_samples}")

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
    valid_data = valid_data.batch(num_train_valid_examples)  # giving a too big of a number = keep in one batch
    test_data = test_data.batch(num_test_examples)  #keep in one batch

    valid_inputs, valid_targets = next(iter(valid_data))  # next will load next batch, since only one, will load everything

    return train_data, valid_inputs, valid_targets, test_data

def prepare_model(in_dic):
    input_size = 784  # 28*28 since we are flattening the image of 28 by 28 pixels
    output_size = 10

    # Add input layer
    model = tf.keras.Sequential([
                                tf.keras.layers.Flatten(input_shape=(28,28,1)),
                                ])

    for i in range(in_dic['Num layers'] - 2):  # add hidden layers
        model.add(tf.keras.layers.Dense(in_dic['Hidden width'], activation=in_dic['Hidden funcs'][i]))

    # Add output layer
    model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=in_dic['Learning rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def single_model(train_data, valid_inputs, valid_targets, test_data, in_dic):
    out_dic = {}

    model = prepare_model(in_dic)

    earlyCallback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=in_dic['Accuracy improvement delta'],
                                                     patience=in_dic['Accuracy improvement patience'],
                                                     restore_best_weights=False)

    start = timer()
    history = model.fit(train_data, epochs=in_dic['Max num epochs'], callbacks=[earlyCallback],
                        validation_data=(valid_inputs, valid_targets), verbose=2)
    end = timer()

    test_loss, test_accuracy = model.evaluate(test_data)

    actual_num_epochs = len(history.history["val_accuracy"])

    out_dic['Product Accuracy'] = round(history.history['val_accuracy'][-1] *
                                        history.history['accuracy'][-1] *
                                        test_accuracy,
                                        4)
    out_dic['Test Accuracy'] = round(test_accuracy, 4)
    out_dic['Product / Time'] = round(out_dic['Product Accuracy'] / (end - start), 4)
    out_dic['Validate Accuracy'] = history.history["val_accuracy"][-1].round(4)
    out_dic['Train Accuracy'] = history.history["accuracy"][-1].round(4)
    out_dic['Train Time'] = round(end - start, 3)
    out_dic['Num epochs'] = actual_num_epochs
    out_dic['Average epoch time'] = round((end - start) / actual_num_epochs, 4)
    out_dic['Accuracy Validate Max'] = max(history.history['val_accuracy']).round(4)
    out_dic['Accuracy Train Max'] = max(history.history["accuracy"]).round(4)

    return out_dic


def do_numerous_loops(num_loops=1, given_dic=None):
    results = []
    if given_dic:
        in_dic = given_dic
        local_accuracy_improvement_deltas = [in_dic['Accuracy improvement delta']]
        local_accuracy_improvement_patiences = [in_dic['Accuracy improvement patience']]
        local_batch_sizes = [in_dic['Batch size']]
        local_hidden_widths = [in_dic['Hidden width']]
        local_nums_layers = [in_dic['Num layers']]
        local_functions = [in_dic['Hidden funcs']]
        local_learning_rates = [in_dic['Learning rate']]
    else:
        in_dic = {'Max num epochs': MAX_NUM_EPOCHS,
                  'Shuffle seed': 100}
        local_accuracy_improvement_deltas = accuracy_improvement_deltas
        local_accuracy_improvement_patiences = accuracy_improvement_patiences
        local_batch_sizes = batch_sizes
        local_hidden_widths = hidden_widths
        local_nums_layers = nums_layers
        local_functions = functions
        local_learning_rates = learning_rates

    num_regressions_without_functions = num_loops * \
                                        len(local_accuracy_improvement_deltas) * \
                                        len(local_accuracy_improvement_patiences) * \
                                        len(local_batch_sizes) * \
                                        len(local_hidden_widths) * \
                                        len(local_nums_layers) * \
                                        len(local_learning_rates)
    print(f'To perform number regressions: {num_regressions_without_functions} with layers: {local_nums_layers} and functions: {local_functions}')


    mnist_data, num_train_valid_examples, num_test_examples = acquire_data()

    # TODO initiate to better values
    best_test = {'Test Accuracy': 0.001}
    best_product = {'Product Accuracy': 0.001}
    best_product_efficiency = {'Product / Time': 0.001}
    best_validate = {'Validate Accuracy': 0.001}
    best_train = {'Train Accuracy': 0.001}

    num_model_trainings = 0
    time_run_started = timer()

    for batch_size in local_batch_sizes:
        in_dic['Batch size'] = batch_size
        if num_loops == 1:
            # to save time not to do it every time if there is only 1 loop
            in_dic['Shuffle seed'] = 1
            train_data, valid_inputs, valid_targets, test_data = prepare_data(mnist_data,
                                                                              num_train_valid_examples,
                                                                              num_test_examples,
                                                                              batch_size,
                                                                              in_dic['Shuffle seed'])

        for accuracy_improvement_delta in local_accuracy_improvement_deltas:
            in_dic['Accuracy improvement delta'] = accuracy_improvement_delta
            for accuracy_improvement_patience in local_accuracy_improvement_patiences:
                in_dic['Accuracy improvement patience'] = accuracy_improvement_patience
                for num_layers in local_nums_layers:
                    in_dic['Num layers'] = num_layers
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
                                accum_test = 0
                                accum_product = 0
                                accum_product_efficiency = 0
                                accum_validate = 0
                                accum_train = 0
                                for loop in range(1, num_loops+1):
                                    if num_loops > 1:
                                        # if more than 1 loops, to allow for different seed, need to prepare data every time
                                        in_dic['Shuffle seed'] = loop
                                        train_data, valid_inputs, valid_targets, test_data = prepare_data(mnist_data,
                                                                                                          num_train_valid_examples,
                                                                                                          num_test_examples,
                                                                                                          batch_size,
                                                                                                          in_dic['Shuffle seed'])

                                    time_running_sec = timer() - time_run_started
                                    print(f'Model {num_model_trainings}, loop {loop}/{num_loops}, '
                                          f'total time min: {round(time_running_sec / 60, 1)}, '
                                          f'total time hours: {round(time_running_sec / 60 / 60, 2)}: '
                                          f'seconds per model: {round(time_running_sec / num_model_trainings)} '
                                          f'====================================')
                                    out_dic = single_model(train_data, valid_inputs, valid_targets, test_data, in_dic)
                                    result = in_dic.copy()
                                    result.update(out_dic)
                                    results.append(result)
                                    print(f'\nCURRENT: {result}')

                                    accum_test += result['Test Accuracy']
                                    accum_product += result['Product Accuracy']
                                    accum_product_efficiency += result['Product / Time']
                                    accum_validate += result['Validate Accuracy']
                                    accum_train += result['Train Accuracy']

                                if (accum_test / num_loops) > best_test['Test Accuracy']:
                                    best_test = {'Average loop result': round(accum_test / num_loops, 4)}
                                    best_test.update(result)
                                if (accum_product / num_loops) > best_product['Product Accuracy']:
                                    best_product = {'Average loop result': round(accum_product / num_loops, 4)}
                                    best_product.update(result)
                                if (accum_product_efficiency / num_loops) > best_product_efficiency['Product / Time']:
                                    best_product_efficiency = {'Average loop result': round(accum_product_efficiency / num_loops, 4)}
                                    best_product_efficiency.update(result)
                                if (accum_validate / num_loops) > best_validate['Validate Accuracy']:
                                    best_validate = {'Average loop result': round(accum_validate / num_loops, 4)}
                                    best_validate.update(result)
                                if (accum_train / num_loops) > best_train['Train Accuracy']:
                                    best_train = {'Average loop result': round(accum_train / num_loops)}
                                    best_train.update(result)

                                print("Finished all loops +++++++++++++++++++++++++++++++++++++++++++++")
                                print(f'BEST TEST ACCURACY:     {best_test}')
                                print(f'BEST PRODUCT ACCURACY:  {best_product}')
                                print(f'BEST EFFICIENT PRODUCT: {best_product_efficiency}')
                                print(f'BEST VALIDATE ACCURACY: {best_validate}')
                                print(f'BEST TRAIN ACCURACY:    {best_train}')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'Total number of models trained: {num_model_trainings} with {num_loops} loops per model')
    pf = pd.DataFrame(results)
    print(f'ALL RESULTS:')
    print(pf.to_string())
    pf.to_excel("output\\full.xlsx")

    time_running_sec = timer() - time_run_started

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
        'Improvement deltas': local_accuracy_improvement_deltas,
        'Improvement delta': local_accuracy_improvement_patiences}

    pf = pd.DataFrame([hyperparams])
    print(f'HYPERPARAMS:')
    print(pf.to_string())
    pf.to_excel("output\\hyperparams.xlsx")

    best_test_with_type = {'Type': 'TEST ACCURACY'}
    best_test_with_type.update(best_test)
    best_product_with_type = {'Type': 'PROD ACCURACY'}
    best_product_with_type.update(best_product)
    best_product_efficiency_with_type = {'Type': 'EFFICIENT PROD'}
    best_product_efficiency_with_type.update(best_product_efficiency)
    best_validate_with_type = {'Type': 'VALIDATE ACCURACY'}
    best_validate_with_type.update(best_validate)
    best_train_with_type = {'Type': 'TRAIN ACCURACY'}
    best_train_with_type.update(best_train)

    pf = pd.DataFrame([best_test_with_type,
                       best_product_with_type,
                       best_product_efficiency_with_type,
                       best_validate_with_type,
                       best_train_with_type])
    print(f'BEST RESULTS:')
    print(pf.to_string())
    pf.to_excel("output\\best.xlsx")


do_numerous_loops(1)
"""
do_numerous_loops(2, {'Accuracy improvement delta': 0.0001,
                      'Accuracy improvement patience': 3,
                      'Max num epochs': 1000,
                      'Batch size': 300,
                      'Num layers': 4,
                      'Hidden funcs': ('tanh', 'relu'),
                      'Hidden width': 200,
                      'Learning rate': 0.001})
"""