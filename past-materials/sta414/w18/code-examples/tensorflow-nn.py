# Fit a Neural Network for classification using TensorFlow

# First, implement the Iris dataset example from the docs
# https://www.tensorflow.org/get_started/get_started_for_beginners#the_iris_classification_problem
# Important: note that as is the case with most of the Tensorflow examples, the code
# that Google posts doesn't actually work or seem to be tested at all. The below is a
# modified, working version

import tensorflow as tf
import numpy as np
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)

# Call load_data() to parse the CSV file.
(train_feature, train_label), (test_feature, test_label) = load_data()

# Create feature columns for all features.
my_feature_columns = [
    tf.feature_column.numeric_column(key='SepalLength'),
    tf.feature_column.numeric_column(key='SepalWidth'),
    tf.feature_column.numeric_column(key='PetalLength'),
    tf.feature_column.numeric_column(key='PetalWidth')
]

# Define the model, using Tensorflow's Estimator.DNNClassifier object
# DNN = Deep Neural Network
nn_basic = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns, # Specify the architecture of the input layer
    hidden_units=[10, 10], # Specify the architecture of the hidden layers
    n_classes=3,
    model_dir='./tmp/nn_basic1') # Specify the architecture of the output layer

# Note the input and output layers are implied by the structure in your data (# of features, # of classes)
# The hidden layer architecture is up to you

# Define the training input function (this terminology is specific to tensorflow)
def train_input_fn(features, labels, batch_size):

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

# Train the model
# Tensorflow saves model checkpoint information to disk
# So you can run the below statement, then run it again, and it will pick up
# training where you left off
nn_basic.train(
	input_fn=lambda: train_input_fn(train_feature, train_label, 120),
	steps=10000)

# Evaluate the model.
def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = dict(features)
    else:
        inputs = (dict(features), labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

eval_result = nn_basic.evaluate(
    input_fn=lambda:eval_input_fn(test_feature, test_label, 120))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# What happens when we change the architecture?
# Go deeper and fatter

nn_deeper = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns, # Specify the architecture of the input layer
    hidden_units=[100, 100, 100], # Specify the architecture of the hidden layers
    n_classes=3) # Specify the architecture of the output layer

nn_deeper.train(
	input_fn=lambda: train_input_fn(train_feature, train_label, 120),
	steps=10000)

eval_result = nn_deeper.evaluate(
    input_fn=lambda:eval_input_fn(test_feature, test_label, 120))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Autoencoder?

nn_autoencoder = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns, # Specify the architecture of the input layer
    hidden_units=[100, 50, 10, 50, 100], # Specify the architecture of the hidden layers
    n_classes=3) # Specify the architecture of the output layer

nn_autoencoder.train(
	input_fn=lambda: train_input_fn(train_feature, train_label, 120),
	steps=10000)

eval_result = nn_autoencoder.evaluate(
    input_fn=lambda:eval_input_fn(test_feature, test_label, 120))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# On our very simple problem (120 training cases, 4 features), changing the
# discrete structure of the network doesn't change the performance.
# What do you think is happening here?

