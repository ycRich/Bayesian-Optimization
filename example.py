from BO import *

def objFunc(x):
	# define objective function to be optimized
	y = x**2 - 5*x
	return  y

X_sample = np.array([[1,4,7]]).T
y_sample = [objFunc(x) for x in X_sample]
print X_sample
print y_sample

X_range = np.array([[0],[10]])
n_points = np.array([20])

x_opt, y_opt, Y_min, U = bayesianOpt(objFunc, X_sample, y_sample, X_range, n_points, acf = "ei", reward = [1, 1, 1])
print x_opt, y_opt

plt.figure()
sns.tsplot(Y_min[1:])
plt.figure()
sns.tsplot(U[1:])
plt.show()