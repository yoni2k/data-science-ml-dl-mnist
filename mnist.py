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
- Change BATCH_SIZE to be dynamic
- Change HIDDEN_WIDTH to be dynamic
- Change NUM_EPOCHS to be dynamic
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

""" Settings """
FRACTION_VALIDATE = 0.1
BATCH_SIZE = 10000
HIDDEN_WIDTH = 50
NUM_EPOCHS = 1

def acquire_data():
    mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

    print(f'Name: {mnist_info.full_name}')
    print(f'Description: {mnist_info.description}')
    print(f'Size in bytes: {mnist_info.size_in_bytes}')
    print(f'Features: {mnist_info.features}')

    print(f"Num train samples: {mnist_info.splits['train'].num_examples}, Num training samples: {mnist_info.splits['test'].num_examples}")

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


def prepare_data(mnist_dataset, num_train_valid_examples, num_test_examples):
    train_valid_data = mnist_dataset['train']
    test_data  = mnist_dataset['test']

    # Scale data
    train_valid_data = train_valid_data.map(scale_0_to_1)
    test_data = test_data.map(scale_0_to_1)

    # Shuffle training data
    train_valid_data = train_valid_data.shuffle(num_train_valid_examples)
    test_data = test_data.shuffle(num_test_examples)

    # Split training and validation data
    valid_data, train_data = split_data(train_valid_data, num_train_valid_examples)

    # Batch train data
    train_data = train_data.batch(BATCH_SIZE)
    valid_data = valid_data.batch(num_train_valid_examples)  # giving a too big of a number = keep in one batch
    test_data = test_data.batch(num_test_examples)  #keep in one batch

    valid_inputs, valid_targets = next(iter(valid_data))  # next will load next batch, since only one, will load everything

    return train_data, valid_inputs, valid_targets, test_data

def prepare_model():
    input_size = 784  # 28*28 since we are flattening the image of 28 by 28 pixels
    output_size = 10

    model = tf.keras.Sequential([
                                tf.keras.layers.Flatten(input_shape=(28,28,1)),
                                tf.keras.layers.Dense(HIDDEN_WIDTH, activation='relu'),
                                tf.keras.layers.Dense(HIDDEN_WIDTH, activation='relu'),
                                tf.keras.layers.Dense(output_size, activation='softmax'),
                                ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def single_model(mnist_data, num_train_valid_examples, num_test_examples, in_dic):
    out_dic = {}

    train_data, valid_inputs, valid_targets, test_test = prepare_data(mnist_data, num_train_valid_examples,
                                                                      num_test_examples)
    model = prepare_model()

    start = timer()
    history = model.fit(train_data, epochs=in_dic['Num epochs'], validation_data=(valid_inputs, valid_targets), verbose=2)
    end = timer()
    print("Time took to fit: " + str(end - start))
    print(history.history)
    print(f'history.history["accuracy"][0]: {history.history["accuracy"][0]}')
    print(f'history.history["val_accuracy"][0]: {history.history["val_accuracy"][0]}')

    out_dic['Train accuracy'] = history.history["accuracy"][0]
    out_dic['Validate accuracy'] = history.history["val_accuracy"][0]
    out_dic['Train time'] = end - start
    out_dic['Average epoch time'] = (end - start) / in_dic['Num epochs']

    return out_dic


def do_numerous_loops():
    results = []
    in_dic = {}

    for num_epochs in [1, 2]:
        in_dic['Num epochs'] = num_epochs
        out_dic = single_model(mnist_data, num_train_valid_examples, num_test_examples, in_dic)
        out_dic.update(in_dic)
        results.append(out_dic)
        print(f'out_dic: {out_dic}')

    print(f'results:\n{results}')
    pf = pd.DataFrame(results)
    print(f'pf: \n{pf}')
    pf.to_csv("output.csv")


mnist_data, num_train_valid_examples, num_test_examples = acquire_data()
do_numerous_loops()
