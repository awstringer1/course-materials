### Implement a single-layer fully connected feed-forward Neural Network
# This example is closely based off of the autograd example, https://github.com/HIPS/autograd/blob/master/examples/neural_net.py

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from data import load_mnist

import argparse
import os
import sys

# Load the MNIST data
# To get this to work, copy the data.py and data_mnist.py files from the autograd/examples github directory
# Then create an empty file in the same folder called __init__.py which has the effect of making .py files in that directory importable
# Then run the above from data import load_mnist line, and you should be good
N, train_images, train_labels, test_images,  test_labels = load_mnist()

# Look at it
N
train_images.shape
train_labels.shape
test_images.shape
test_labels.shape

# Add a column of 1s and subsample (60,000 obs with 784 features and 30 hidden layers makes the gradient take about 10 seconds to compute each time, which is slow)
train_subsample = np.random.choice(train_images.shape[0],6000,False)
test_subsample = np.random.choice(test_images.shape[0],1000,False)

train_images1 = np.array([np.concatenate(([1.],x)) for i, x in enumerate(train_images) if i in train_subsample])
test_images1 = np.array([np.concatenate(([1.],x)) for i, x in enumerate(test_images) if i in test_subsample])
train_labels1 = np.array([x for i, x in enumerate(train_labels) if i in train_subsample])
test_labels1 = np.array([x for i, x in enumerate(test_labels) if i in test_subsample])


train_images1.shape
test_images1.shape

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Function for prediction
def neural_net_predict(params, inputs,debug=False):
    """Implements a neural network for classification.
       params is a list of (weights) tuples.
       inputs is an (N x D+1) matrix, the first column of which is a column of 1s (hence we don't explicitly 
       consider the bias term)
       returns normalized class probabilities."""
    i = 1
    for W in params:
        outputs = np.dot(inputs, W) # Note: this implementation is vectorized over inputs
        inputs = sigmoid(outputs)
        if debug:
	        print("Neural Net Predict: Layer {}, outputs shape = {}, inputs shape = {}".format(i,outputs.shape,inputs.shape))
	        i+=1
    return np.array(tuple([x / np.sum(x,axis=0) for x in sigmoid(outputs)]))

# Measure accuracy
def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

# Compute loss function
def loss_function(params, inputs, targets):
    return -np.sum(np.log(neural_net_predict(params, inputs,debug=False)) * targets)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--learningrate', nargs='?', const=True, type=float,
	              default=0.01,
	              help='Learning rate for gradient descent')
	parser.add_argument('--numiter', nargs='?', const=True, type=float,
	              default=100,
	              help='Number of iterations of gradient descent')
	FLAGS, unparsed = parser.parse_known_args()

	# Set the layer size
	layer_size = 30 # 30 Hidden units
	input_size = train_images1.shape[1] # Number of features
	output_size = train_labels1.shape[1] # Number of classes

	# Set the number of gradient descent iterations
	numiter = FLAGS.numiter
	# Set the learning rate
	learning_rate = FLAGS.learningrate

	# Initialize parameters
	init_params = [(np.random.RandomState(0).randn(input_size,layer_size)),
					(np.random.RandomState(0).randn(layer_size,output_size)),
	]

	# Objective function is loss_function. Get its gradient with respect to params
	objective_gradient = grad(loss_function)
	# Note that params is a list of matrices (2d numpy arrays), and hence so is the gradient

	# Implement gradient descent. We're updating each row of each element of the list of parameters separately; this has nothing to do with gradient
	# descent, it's just how we chose to store the parameters. You could flatten them out and store the whole thing as a 1d array, which would agree
	# with the notation used in class and in the math behind this.
	params = init_params
	iter = 1
	while iter <= numiter:
		# Compute the gradient once
		the_gradient = objective_gradient(params,train_images1,train_labels1)
		# Loop over the list of matrices, updating each set of parameters
		for m in range(len(the_gradient)):
			for r in range(len(params[m])):
				params[m][r] -= learning_rate * the_gradient[m][r]


		# Print the train and test accuracy
		if iter % 10 == 0:
			print("Iteration {} of Gradient Descent. Training accuracy: {}, Test accuracy: {}".format(iter,accuracy(params,train_images1,train_labels1),accuracy(params,test_images1,test_labels1)))

		iter += 1

print("After {} iterations of Gradient Descent. Training accuracy: {}, Test accuracy: {}".format(iter,accuracy(params,train_images1,train_labels1),accuracy(params,test_images1,test_labels1)))




















