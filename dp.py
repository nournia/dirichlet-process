from math import *
import numpy as np
# np.random.seed(1)

# laad data
file = open('students.dat')
data = []
for line in file:
	data.append(int(line))

# data
Y = np.array(data)
# labels
X = len(Y) * [int(round(Y.mean()))]

# parameters
muG, sigmaG = Y.mean(), Y.std() # G0 ~ N(muG, sigmaG) # best guess of G
A0 = 0.001 # strength of belief that G0 is G

from scipy.stats import norm
def phi(d): # value of d in normal distribution function
	return norm.pdf(d)
def A(y):
	muF, sigmaF = 0, 1
	sigmaSum = sigmaF**2 + sigmaG**2
	return A0 * exp(-1 * (y - (muF + muG))**2 / (2*sigmaSum)) / sqrt(2*pi*sigmaSum)

# mu0: mean of current cluster
def getLabelOfNewCluster(y, mu0):
	return int(round(np.random.normal((mu0 + y)/2, 2**-.5))) # sample Gausian distribution

# make cluster dictionary from data and labels
clusters = {}
def appendToCluster(i):
	global clusters
	if not clusters.has_key(X[i]): clusters[X[i]] = []
	clusters[X[i]].append(Y[i])

for i in range(len(Y)):
	appendToCluster(i)

# iteration
iterations = 100
for itr in range(1, iterations+1):
	#update temperature
	T = np.random.uniform(iterations-itr+1, iterations-itr+1+10)
	powT = 3*(10+iterations - T)/(10+iterations)

	# update X[i] in n+1 way using the multinomial selector
	for i in range(len(Y)):
		probabilites = [] # of each condition
		for label, items in clusters.iteritems():
			probabilites.append(phi(Y[i] - label) * len(items))
		probabilites.append(A(Y[i]))
		probabilites = np.array(probabilites)
		
		# simulated annealing effect
		probabilites = probabilites**powT

		# sum of probabilites equals to 1
		probabilites /= probabilites.sum()
		
		# sample from multinomial distribution
		choice = np.random.multinomial(1, probabilites).tolist().index(1) # search the 1 element's index

		label = X[i]
		if choice == len(clusters): # add new cluster
			X[i] = getLabelOfNewCluster(Y[i], np.array(clusters[X[i]]).mean())
		else: # change cluster
			X[i] = clusters.keys()[choice]

		# update clusters
		if label != X[i]:
			clusters[label].remove(Y[i])
			if len(clusters[label]) == 0: clusters.pop(label)
			appendToCluster(i)

	# print results
	for label in sorted(clusters.iterkeys()):
		print '{} -> {}'.format(label, clusters[label])
	print 'iteration: {}, clusters: {}, temperature: {:.2f}'.format(itr, len(clusters), T)
	print