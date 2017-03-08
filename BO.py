'''
Beyesian Optimization
Created by Yichi Zhang
2017-Feb-27
'''
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

np.random.seed(1234)


def acquisitionFunc(X_pred, y_pred, sigma, y_min, acf = 'ei', reward = [0.33,0.33,0.33]):
	# calculate expected improvement for selecting next sampling point
	# X_pred - (n_samples, n_features), numpy array
	# y_pred - (n_samples, 1), numpy array
	# sigma - (n_samples,), numpy array
	# y_min - scalar
	if acf == 'ei':
		beta = [(y_min - y_pred[i]) / sigma[i] if sigma[i] > 0 else 0 for i in range(len(sigma))]
		ei = [(y_min - y_pred[i])*norm.cdf(beta[i])+ sigma[i]*norm.pdf(beta[i]) for i in range(len(sigma))]
		x_next = X_pred[np.argmax(ei),]
		utility = max(ei)

	elif acf == 'lcb':
		lcb = [2*sigma[i] - y_pred[i] for i in range(len(y_pred))]
		x_next = X_pred[np.argmax(lcb),]
		utility = max(lcb)
	elif acf == 'pi':
		beta = [(y_min - y_pred[i] - 0.01) / sigma[i] if sigma[i] > 0 else 0 for i in range(len(sigma))]
		pi = norm.cdf(beta)
		x_next = X_pred[np.argmax(pi),]
		utility = max(pi)
	elif acf == 'gpHedge':
		r = np.random.rand()
		total_reward = sum(reward)

		if r < reward[0] / total_reward:
			acf = 'ei'
		elif r < (reward[0] + reward[1]) / total_reward:
			acf = 'lcb'
		else:
			acf = 'pi'
		x_next, utility, acf = acquisitionFunc(X_pred, y_pred, sigma, y_min, acf)

	return x_next, utility, acf


def gridSamp(X_range, n_points):
	# X_range - (2, n_features) numpy array
	# n_points - (n_features, ) numpy array
	n = X_range.shape[1]
	S = []

	if n > 1:
		
		A = gridSamp(X_range[:,1:], n_points[1:])
		m = len(A)
		q = n_points[0]
		
		S = np.concatenate((np.zeros((m*q,1)), np.matlib.repmat(A, q, 1)), axis = 1)
		
		y = np.linspace(X_range[0,0], X_range[1,0], q)
		k = range(m)
		
		for i in range(q):
			
			S[k, 0] = np.matlib.repmat(y[i], 1, m)
			k = [x + m for x in k]
	else:
		S = np.linspace(X_range[0,0],X_range[1,0], n_points)[np.newaxis].T

	return S


def bayesianOpt(objFunc, X_sample, y_sample, X_range, n_points = 20, n_restarts_optimizer = 10, acf = 'ei', reward = [0.33, 0.33, 0.33]):
	# X_range is a 2 by n_features array, first row is the lower bound, second row is the upper bound
	y_min_old = float('Inf')
	y_min = min(y_sample)
	U, Y_min = np.array([1e-2]), np.array([y_min])
	reward_history = np.array([reward])
	
	# loop until stopping criteria satisfied
	while (abs(y_min - y_min_old) > 1e-8): 
		
		y_min_old = y_min
		# gp model from sklearn
		gp = GaussianProcessRegressor(n_restarts_optimizer = n_restarts_optimizer) 
		gp.fit(X_sample, y_sample)

		# create meshgrid for prediction
		X_pred = gridSamp(X_range, n_points) 
		# make prediction
		y_pred, sigma = gp.predict(X_pred, return_std = True) 
		
		# finding the optimum
		y_min = float('Inf')
		x_min = []
		for i in range(10): 
			# generate random starting point multiple times to avoid local optimum
			x0 = np.multiply(np.random.rand(1, X_range.shape[1]), X_range[1,:] - X_range[0,:])
			# use scipy minimizer to find min of current model
			res = minimize(lambda x: gp.predict(np.array([x])), x0)
			x_opt, y_opt = res.x, res.fun
			if y_opt < y_min:
				x_min, y_min = x_opt, y_opt

		# calculate utility and find the next sampling point
		x_next, utility, acf_used = acquisitionFunc(X_pred,y_pred, sigma, y_min, acf = acf, reward = reward)
		y_next = objFunc(x_next)
		
		r = y_min_old - y_min
		if acf_used == 'ei':
			reward[0] += max(r, 0)
		elif acf_used == 'lcb':
			reward[1] += max(r, 0)
		elif acf_used == 'pi':
			reward[2] += max(r, 0)

		reward_history = np.append(reward_history, reward)
		X_sample, y_sample = np.append(X_sample, [x_next], axis = 0), np.append(y_sample, [y_next], axis = 0)

		Y_min = np.append(Y_min, y_min)
		U = np.append(U, utility)


	return x_min, y_min, Y_min, U
