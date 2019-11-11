"""
Gradient Descent example in Python
Written by Alex Stringer for STA414/2104
January 2018

This shows an example of using gradient descent to minimize a loss function
for some training data. The loss function will be the squared-error loss as
defined for linear least squares, except now the prediction function y(x,w)
will be a non-linear function of w with no closed-form gradient-based
optimizer.

We'll generate an example dataset from the model 

y(x,w) = (w[0] + w[1]*sin(2*pi*x)) / (1 + w[2]*x), and

use gradient descent to recover the true parameters w = (1,2,4).
Note: we generate the data with noise, so we won't recover exactly (1,2,4)
"""

# Import required libraries
import autograd.numpy as np # Autograd wrapper of Numpy
from autograd import grad, jacobian
from matplotlib import pyplot as plt


from datetime import datetime


# Set the true value of w
true_w = np.array([1.,2.,4.])

# Prediction function
def y(x,w):
	return (w[0] + w[1]*np.sin(2. * np.pi * x)) / (1 + w[2]*x)

# Loss function
def loss_function(w,dat):
	"""
	Computes the loss function for an entire dataset dat
	Assumes a two-column input, where dat[:,0] == x and dat[:,1] == y
	Note: we make w the first argument so we can easily define the gradient using autograd
	"""
	return 0.5 * np.sum(np.power(y(dat[:,0],w) - dat[:,1],2))

# Autograd gradient and hessian
loss_gradient = grad(loss_function)
loss_hessian = jacobian(loss_gradient)


# Generate data from the model
def generate_data(n):
	"""
	Return a numpy array of dimension n x 2 containing samples from the model
	"""

	x = np.random.uniform(size=n)
	return np.stack((x,y(x,true_w) + np.random.normal(scale=0.1,size=n)),axis=1)

# Generate a dataset from the model
dat = generate_data(n=30)

# Plot it- if you're running this interactively
plt.scatter(x=dat[:,0],y=dat[:,1])
# Add the true curve
plt.plot(np.linspace(0,1,200),y(np.linspace(0,1,200),true_w))
plt.show()

# Note: we plotted y vs x, but our loss function is a function of w = (w[0],w[1],w[2]), 
# and that's what we're optimizing for.

# Gradient descent for w, with a fixed step size

# Initial w
w = np.array([0.,0.,0.])
step = 0.01
t = 0

starttime = datetime.now()

while np.linalg.norm(loss_gradient(w,dat)) > 0.0001:
	w = w - step * loss_gradient(w,dat)
	t = t + 1
	if t % 100 == 0:
		print("At iteration {}, loss function is {} and norm of gradient is {}".format(t,loss_function(w,dat),np.linalg.norm(loss_gradient(w,dat))))

print("Final loss function: {}".format(loss_function(w,dat)))
print("Final parameter estimate: {}".format(w))
print("True parameter estimate: {}".format(true_w))

endtime = datetime.now()
print("{} iterations of gradient descent took {}".format(t,endtime - starttime))

# Newton's method- very sensitive to starting values
w = np.array([0.5,1.3,3.7])
t = 0

starttime = datetime.now()

while np.linalg.norm(loss_gradient(w,dat)) > 0.0001:
	w = w - np.dot(np.linalg.inv(loss_hessian(w,dat)),loss_gradient(w,dat))
	t = t + 1
	if t % 100 == 0:
		print("At iteration {}, loss function is {} and norm of gradient is {}".format(t,loss_function(w,dat),np.linalg.norm(loss_gradient(w,dat))))

print("Final loss function: {}".format(loss_function(w,dat)))
print("Final parameter estimate: {}".format(w))
print("True parameter estimate: {}".format(true_w))

endtime = datetime.now()
print("{} iterations of Newton's Method took {}".format(t,endtime - starttime))



