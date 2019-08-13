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
- Change NUM_EPOCHS to stop after a while
- Change number of Widths (hidden layers)
- Change activation functions (add sigmoid, tanh)
- Give different learning rates
- Do one run of the best inputs

"""

""" Imports """
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import tensorflow as tf
import tensorflow_datasets as tfds  # getting MNIST from there
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from pprint import pprint

""" Settings """
FRACTION_VALIDATE = 0.1
ACCURACY_IMPROVEMENT_DELTA = 0.01  # TODO: change to 0.001 or 0.0001?
ACCURACY_IMPROVEMENT_PATIENCE = 2  # TODO: change to 3?
MAX_NUM_EPOCHS = 15

batch_sizes = [100]
hidden_widths = [50]


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


def prepare_data(mnist_dataset, num_train_valid_examples, num_test_examples, batch_size):
    train_valid_data = mnist_dataset['train']
    test_data = mnist_dataset['test']

    # Scale data
    train_valid_data = train_valid_data.map(scale_0_to_1)
    test_data = test_data.map(scale_0_to_1)

    # Shuffle training data
    train_valid_data = train_valid_data.shuffle(num_train_valid_examples, seed=100)
    test_data = test_data.shuffle(num_test_examples, seed=100)

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

    model = tf.keras.Sequential([
                                tf.keras.layers.Flatten(input_shape=(28,28,1)),
                                tf.keras.layers.Dense(in_dic['Hidden width'], activation='relu'),
                                tf.keras.layers.Dense(in_dic['Hidden width'], activation='relu'),
                                tf.keras.layers.Dense(output_size, activation='softmax'),
                                ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def single_model(train_data, valid_inputs, valid_targets, in_dic):
    out_dic = {}

    model = prepare_model(in_dic)

    earlyCallback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=ACCURACY_IMPROVEMENT_DELTA,
                                                     patience=ACCURACY_IMPROVEMENT_PATIENCE,
                                                     restore_best_weights=True)

    start = timer()
    history = model.fit(train_data, epochs=MAX_NUM_EPOCHS, callbacks=[earlyCallback],
                        validation_data=(valid_inputs, valid_targets), verbose=2)
    end = timer()

    actual_num_epochs = len(history.history["val_accuracy"])

    out_dic['Accuracy Validate Last'] = history.history["val_accuracy"][-1].round(4)
    out_dic['Accuracy Validate Best'] = max(history.history['val_accuracy']).round(4)
    out_dic['Train time'] = round(end - start, 3)
    out_dic['Average epoch time'] = round((end - start) / actual_num_epochs, 4)
    out_dic['Accuracy Train Last'] = history.history["accuracy"][-1].round(4)
    out_dic['Accuracy Train Best'] = max(history.history["accuracy"]).round(4)

    return out_dic


def do_numerous_loops():
    results = []
    in_dic = {}

    for batch_size in batch_sizes:
        in_dic['Batch size'] = batch_size
        train_data, valid_inputs, valid_targets, test_test = prepare_data(mnist_data, num_train_valid_examples,
                                                                          num_test_examples, batch_size)
        for hidden_width in hidden_widths:
            in_dic['Hidden width'] = hidden_width
            out_dic = single_model(train_data, valid_inputs, valid_targets, in_dic)
            result = in_dic.copy()
            result.update(out_dic)
            results.append(result)
            print(result)

    pf = pd.DataFrame(results)
    print(f'Results:')
    print(pf.to_string())
    pf.to_csv("output.csv")


mnist_data, num_train_valid_examples, num_test_examples = acquire_data()
do_numerous_loops()
