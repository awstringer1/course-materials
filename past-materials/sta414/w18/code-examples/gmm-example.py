# Gaussian Mixture Model with 3 groups, fit to the Iris dataset

import numpy as np
from sklearn.datasets import load_iris
import scipy.stats as stats
from datetime import datetime
from scipy.special import logsumexp
from matplotlib import pyplot as plt
%matplotlib qt

# Load the data
iris = load_iris()
iris_data = iris.data



# Function to compute the multivariate Gaussian log-likelihood
def mvn_ll(X,mu,sigma):
	"""
	Compute the log-density for each row of X, and sum
	Arguments:
		X: numpy array where rows represent observations and columns represent features
		mu: numpy array representing the mean vector
		sigma: numpy array representing the covariance matrix
	"""

	ll = 0.0
	n = np.float(X.shape[0])
	p = np.float(X.shape[1])

	for i in range(int(n)):
		ll += stats.multivariate_normal.logpdf(x = X[i],mean = mu,cov = sigma)

	return ll


# irismean = iris_data.mean(axis=0)
# iriscov = np.cov(iris_data,rowvar=False)
# mvn_ll(iris_data,irismean,iriscov)

# Function to compute the marginal log-likelihood for a GMM
def marginal_ll(pivec,muvec,sigmavec,X,K):
	"""
	Compute the marginal loglikelihood for the gaussian mixture model
	Arguments:
		pivec: numpy array of dimension K containing group membership probabilities
		muvec: numpy array of dimension p x K; kth column is mean vector for Kth group
		sigmavec: numpy array of dimension p x p x K; kth array is the covariance matrix for the kth group
		X: numpy array containing the data
		K: number of groups

	Returns:
		single number representing the log likelihood
	"""

	ll = 0.0
	n = X.shape[0]

	for i in range(n):
		# Create a vector of log-likelihoods
		for k in range(K):
			if k == 0:
				llvec = np.array([stats.multivariate_normal.logpdf(X[i],muvec[:,k],sigmavec[k])])
			else:
				llvec = np.concatenate((llvec,np.array([stats.multivariate_normal.logpdf(X[i],muvec[:,k],sigmavec[k],)])))

		# Numerically stable sum of b*exp(a) from scipy
		ll += logsumexp(a = llvec,b = pivec)

	return ll


# E-Step: compute the expected group memberships given the data and current parameter estimates
def estep(pivec,muvec,sigmavec,X,K):
	"""
	Compute n x K matrix of expected group memberships. Rows represent observations,
	columns represent groups

	Arguments:
		pivec: numpy array of dimension K containing group membership probabilities
		muvec: numpy array of dimension p x K; kth column is mean vector for Kth group
		sigmavec: numpy array of dimension p x p x K; kth array is the covariance matrix for the kth group
		X: numpy array containing the data
		K: number of groups

	Returns:
		n x K matrix containing the current expected group memberships
	"""

	n = X.shape[0]
	p = np.float(X.shape[1])

	for i in range(n):
		for k in range(K):
			if k == 0:
				zrow = np.array([pivec[k] * stats.multivariate_normal.pdf(X[i],muvec[:,k],sigmavec[k,:,:])])
			else:
				znew = np.array([pivec[k] * stats.multivariate_normal.pdf(X[i],muvec[:,k],sigmavec[k,:,:])])
				#print("k = {}".format(k))
				#print("Dimension of zrow = {}, dimension of znew = {}".format(zrow.shape,znew.shape))
				zrow = np.concatenate((zrow,znew))

		zrow = zrow / np.sum(zrow)

		if i == 0:
			Z = [zrow]
		else:
			Z += [zrow]

	return np.vstack(tuple(Z))


# Compute one full step of the EM algorithm
def emstep(pivec,muvec,sigmavec,X,K):
	"""
	Compute one full step of the EM Algorithm
	Arguments:
		pivec: numpy array of dimension K-1 containing group memberships for all but the last group
		muvec: numpy array of dimension p x K; kth column is mean vector for Kth group
		sigmavec: numpy array of dimension p x p x K; kth array is the covariance matrix for the kth group
		X: numpy array containing the data

	Returns:
		Updated parameter estimates; a list with components pivec, muvec, and sigmavec
	"""

	# E step
	Z = estep(pivec,muvec,sigmavec,X,K)

	n = X.shape[0]

	zsum = np.sum(Z,axis=0)

	# Update pivec
	pivec = np.mean(Z,axis=0)

	# Update muvec and sigmavec
	for k in range(K):
		# mu needs to be done before sigma
		for i in range(n):
			if i == 0:
				tmpmu = Z[i,k] * X[i] / zsum[k]
			else:
				tmpmu = tmpmu + Z[i,k] * X[i] / zsum[k]

		# Now do sigma
		for i in range(n):
			if i == 0:
				tmpsigma = Z[i,k] * np.outer(X[i] - tmpmu,X[i] - tmpmu) / zsum[k]
			else:
				tmpsigma = tmpsigma + Z[i,k] * np.outer(X[i] - tmpmu,X[i] - tmpmu) / zsum[k]

		if k == 0:
			muvec = [tmpmu]
			sigmavec = [tmpsigma]
		else:
			muvec += [tmpmu]
			sigmavec += [tmpsigma]

	return [pivec,np.stack(tuple(muvec),axis=1),np.stack(tuple(sigmavec),axis=0)]


# Run the EM algorithm on the iris dataset

# Fit a K-group model
K = 3

# Initialize parameters- randomly split the data in K and calculate the sample statistics
n = iris_data.shape[0]
p = iris_data.shape[1]


idx = np.array([int(x) for x in np.random.choice(np.linspace(0,n-1,n),n)])
datasplit = [iris_data[idx[int(n/K)*i:int(n/K)*(i+1)]] for i in range(K)]

datasplit = [iris_data[idx[0:50]],iris_data[idx[50:100]],iris_data[idx[100:150]]]


pivec = np.array([1.0 / float(K) for x in range(K)])
muvec = np.stack([x.mean(axis=0) for x in datasplit],axis=1)
sigmavec = np.stack([np.cov(x,rowvar=False) for x in datasplit],axis=0)

# Should we plot as we go?
plot_it = True


if plot_it:
	# Create plot
	plt.ion()
	fig, ax = plt.subplots()
	sc = ax.scatter(x=iris_data[:,1],y=iris_data[:,2],c=iris.target,cmap='brg')
	#plt.xlim(4,8)
	#plt.ylim(0,8)

	plt.draw()

	plt.waitforbuttonpress()

# Run the algorithm
print("Beginning EM algorithm. System time: {}".format(datetime.now()))

converged = False # Will converge if log-likelihood doesn't increase past a point on an iteration

itr = 1

oldloglik = 0.0
loglik = marginal_ll(pivec,muvec,sigmavec,iris_data,K)

while not converged:

	if plot_it:
		# Update plot with current classification as colour
		Z = estep(pivec,muvec,sigmavec,iris_data,K)
		whichzmax = np.argmax(Z,axis=1)
		sc = ax.scatter(x=iris_data[:,1],y=iris_data[:,2],c=whichzmax,cmap='brg')
		plt.draw()
		plt.waitforbuttonpress()

	# Update parameters
	pivec, muvec, sigmavec = emstep(pivec,muvec,sigmavec,iris_data,K)
	# Calculate loglikelihood
	oldloglik = loglik
	loglik = marginal_ll(pivec,muvec,sigmavec,iris_data,K)

	# Check convergence
	if abs(loglik - oldloglik) < 0.0001 or itr > 200:
		converged = True

	print("Iteration {} of EM Algorithm. Marginal log-likelihood: {}".format(itr,loglik))
	itr += 1


# Compute final soft classifications
Z = estep(pivec,muvec,sigmavec,iris_data,K)

print("EM Algorithm Completed. System time: {}".format(datetime.now()))
print("EM Algorithm took {} iterations. Final marginal log likelihood: {}".format(itr,loglik))
