from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import *

def guassianPlot2D(mu,sigma):
	mu = np.array(mu).reshape((-1,1))
	sigma = np.array(sigma).reshape(2,2)
	assert(len(mu)==2)

	# first we generate the unit circle of (x,y) points
	def PointsInCircum(r,n=100):
		return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]
	pts = np.array(PointsInCircum(2)).T  # 2xN

	# we then calculate the sqrt of the sigma
	# the np.eig has output ( (N,), (N,N) )
	# where each col of eig_vec is eigen vector and is of norm=1
	# note that eig_val is not sorted
	# eig_vec * eig_val * eig_vec.T = sigma
	eig_val, eig_vec = eig(sigma)  # we assume sigma is positive definite, so eig_val > 0
	eig_val_sqrt = np.sqrt(eig_val).reshape((1,-1))  # 1x2
	sigma_sqrt = eig_vec*eig_val_sqrt  # 2x2

	# finally, transform pts based on sigma_sqrt
	# y = Ax, cov(y) = A*cov(x)*A.T
	# since cov(x) = I, so cov(y) = A*A.T
	# if we let A = sqrt(sigma), then cov(y) = sigma, which is the covariance matrix we need
	pts = np.matmul(sigma_sqrt,pts)  # 2xN
	pts += mu

	return pts

# Assume X is of size DxN
# Return an (N,) array
def calGassuianProb_cov(X,mu,cov):
	mu = np.array(mu).reshape((-1,1))  # Dx1
	D = len(mu)
	cov = np.array(cov).reshape(D,D)
	X_mu = X - mu  # DxN
	return np.exp ( - (np.log(np.linalg.det(cov)) + np.diag(np.matmul(np.matmul(X_mu.T,inv(cov)),X_mu)) + D*np.log(2*np.pi)) / 2.0 )

def calGassuianProb(X,mu,prec):
	mu = np.array(mu).reshape((-1,1))  # Dx1
	D = len(mu)
	prec = np.array(prec).reshape(D,D)
	X_mu = X - mu  # DxN
	return np.exp ( (np.log(np.linalg.det(prec)) - np.diag(np.matmul(np.matmul(X_mu.T,prec),X_mu)) - D*np.log(2*np.pi) ) / 2.0 )

def calLogGassuianProb(X,mu,prec):
	mu = np.array(mu).reshape((-1,1))  # Dx1
	D = len(mu)
	prec = np.array(prec).reshape(D,D)
	X_mu = X - mu  # DxN
	return (np.log(np.linalg.det(prec)) - np.diag(np.matmul(np.matmul(X_mu.T,prec),X_mu)) - D*np.log(2*np.pi) ) / 2.0

# Sampling N points from Gaussian
def sampleByGaussian(mu,cov,N):
	mu = np.array(mu).reshape([-1,1])
	dim = len(mu)
	cov = np.array(cov).reshape([dim,dim])
	L = cholesky(cov)	# LL'=cov(y)
	# Let y=Lx, yy'=Lxx'L'=Lcov(x)L'=LL'=cov(y)
	unitNormal = np.random.normal(size=[dim*N]).reshape([dim,N])
	X = np.matmul(L,unitNormal) + mu
	return X

# Calculate y, such that exp(y) = exp(x1) + exp(x2)
# In most cases, exp(x) is used for probability, which means 0~1
# and if (x1-x2)>20, exp(x2) is almost 0 (beyond the precision capibility)
# so setting th=20 is meaningful, and maybe 10 is enough
def logAdd(x1,x2):
	th = 20.0
	if x2>x1:
		x1, x2 = x2, x1
	# here we assume x1>x2
	if (x1-x2)>th:
		return x1
	else:
		return x1 + np.log(1+np.exp(x2-x1))

def logAddVec(xvec):
	y = xvec[0]
	for x in xvec[1:]:
		y = logAdd(y,x)
	return y

# Input:
#   K: mixutre number
#   p: dim
#   T: number of samples
#   priorRatio: scalar (0.0~1.0)
#   gmm_w: (K,1)
#   gmm_mu: (K,p)
#   gmm_cov: (K,p,p), should be the same shape with gmm['prec']
# Return:
#   v, tau, alpha: (K,1)
#   mu: (p,K)
#   sigma: (K,p,p)
def initHyperParams(K,p,T,priorRatio,gmm_w,gmm_mu,gmm_cov):
  pseudo_num_ubm = priorRatio*T/(1.0-priorRatio)  # can be seen as the pseudo number of data that was used for trainin the UBM
  # The parameters for Dirichlet distribution
  v = np.ones([K,1]) + gmm_w*pseudo_num_ubm # this initialization is important, to avoid eq (21) fails when sum(BETA[k,:])=0
  # The parameters for Normal-Wishart distribution
  tau = np.ones([K,1])*pseudo_num_ubm/K # (K,1), assign the pseudo number for each mixtures equally
  # tau = pseudo_num_ubm*gmm['w'] # (K,1), assign the pseudo number for each mixtures according to the mixture weights
  # The parameters for Normal-Wishart distribution
  alpha = np.ones([K,1])*(p+1)  # alpha should > p-1 for well definition. We set p+1 to avoicd eq (23) fails as well as keep well defined
  # The parameters for Normal-Wishart distribution
  mu = np.copy(gmm_mu.reshape([K,p]).T)  # (p,K), initialized with UBM's means
  # The parameters for Normal-Wishart distribution
  sigma = np.copy(gmm_cov) # (K,p,p), initialized with UBM's covariance matrices
  return v, tau, alpha, mu, sigma