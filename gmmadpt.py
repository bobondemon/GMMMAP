import matplotlib.pyplot as plt
import numpy as np
from numpy import log, exp, matmul
from helper import *
from numpy.linalg import *

'''
This is a python implementation of the paper, which only implemented GMM part
"Maximum A Posteriori Estimation for Multivariate Gaussian Mixture Observations of Markov Chains"
We keep the notations as much as the same in the paper
'''

#	===== Input Argument Foramt
# gmm has format
# 	gmm['mu']: (K,p)
# 	gmm['prec']: (K,p,p)
# 	gmm['cov']: (K,p,p)
# 	gmm['w']: (K,)
# X has format
# 	(p, T), where p is dimension, T is number of observations
# priorRatio is a scalar
# 	representing the data number precentage of UBM
# 	e.g. if T=500, and priorRatio=0.25, then we assume we have tau = (500/0.75)*0.25
def gmmMap(in_gmm,X,priorRatio,maxItrNum=10,adptW=False,adptM=True,adptCov=False,hyper=None):
	eps = np.finfo(float).eps
	X = np.array(X)	# (p,T)
	assert(X.ndim==2)
	p, T = X.shape

	gmm = {}
	gmm['mu'] = np.array(in_gmm['mu'])
	gmm['prec'] = np.array(in_gmm['prec'])
	gmm['cov'] = np.array(in_gmm['cov'])
	gmm['w'] = np.array(in_gmm['w'])

	K = len(gmm['w'])
	gmm['w'] = gmm['w'].reshape([K,1])	# (K,1)

	assert(gmm['mu'].ndim==2 and gmm['prec'].ndim==3 and gmm['cov'].ndim==3)
	assert(K==len(gmm['mu']) and K==len(gmm['prec']) and K==len(gmm['cov']))
	assert(p==gmm['mu'].shape[1] and p==gmm['prec'].shape[1] and p==gmm['prec'].shape[2] and p==gmm['cov'].shape[1] and p==gmm['cov'].shape[2])
		
	# ========== Init prior hyper-parameters
	#   v, tau, alpha: (K,1)
	#   mu: (p,K)
	#   sigma: (K,p,p)
	if not hyper:
		v, tau, alpha, mu, sigma = initHyperParams(K,p,T,priorRatio,gmm['w'],gmm['mu'],gmm['cov'])
	else:
		print('Using the input hyper-parameters')
		alpha, mu, sigma = hyper['alpha'], hyper['mu'], hyper['sigma']
		pseudo_num_ubm = priorRatio*T/(1.0-priorRatio)  # can be seen as the pseudo number of data that was used for trainin the UBM
		v = np.ones([K,1]) + gmm['w']*pseudo_num_ubm # this initialization is important, to avoid eq (21) fails when sum(BETA[k,:])=0
		tau = np.ones([K,1])*pseudo_num_ubm/K # (K,1), assign the pseudo number for each mixtures equally

	# ========== EM Iteration
	for itr in range(maxItrNum):
		# ===== E-Step
		# logP[k,t] is the log-probability of X(:,t) to the k-th Gaussian, (not included the weight of the mixture)
		logP = np.zeros([K,T])
		for k in range(K):
			logP[k,:] = calLogGassuianProb(X,gmm['mu'][k],gmm['prec'][k])
		logW = np.log(gmm['w'])
		logPW = logP + logW  # (K,T)
		logSumPW = np.zeros([1,T])  # (1,T)
		for i in range(T):
			logSumPW[0,i] = logAddVec(logPW[:,i])

		logBETA = logPW - logSumPW	# (K,T)
		BETA = exp(logBETA)	# (K,T)
		# [Debug]
		# check the sum(BETA,axis=0) to see if values are all close to 1.0
		# print(np.sum(BETA,axis=0))
		# [END Debug]
		sumBETA = np.sum(BETA,axis=1).reshape([K,1])  # (K,1)
		# print(sumBETA)
		xbar = np.zeros([p,K])	# (p,K)
		xGlobalMean = np.sum(X,axis=1)  # (p,)
		xBarWeightedByBeta = np.zeros([p,K])	# (p,K)
		S = np.zeros([K,p,p])
		for k in range(K):
			xBarWeightedByBeta[:,k] = np.sum(BETA[k,:].reshape([1,T])*X,axis=1)
			if sumBETA[k]<eps:
				# print('sumBETA[k]=0')
				xbar[:,k] = xGlobalMean
			else:
				xbar[:,k] = xBarWeightedByBeta[:,k]/sumBETA[k]
			datatmp = X - xbar[:,k].reshape([p,1])	# (p,T)
			S[k,...] = np.matmul(BETA[k,:].reshape([1,T])*datatmp,datatmp.T)

		# ===== M-Step
		updatedW = (v-1+sumBETA)/np.sum(v-1+sumBETA)	# (K,1)
		if adptW:
			gmm['w'] = updatedW
		
		numerator = (tau.T*mu + xBarWeightedByBeta)	# (p,K)
		denominator = (tau+sumBETA).T	# (1,K)
		updatedM = numerator/denominator	# (p,K)
		if adptM:
			gmm['mu'] = (updatedM.T).reshape([K,p])	# (K,p,)

		updatedCov = np.zeros([K,p,p])
		for k in range(K):
			X_minus_mean = X-updatedM[:,k].reshape([p,1])	# (p,T)
			mu_minus_mean = mu[:,k].reshape([p,1])-updatedM[:,k].reshape([p,1])	# (p,1)
			numerator = sigma[k] + np.matmul(BETA[k,:].reshape([1,T])*X_minus_mean,X_minus_mean.T) + tau[k]*np.matmul(mu_minus_mean, mu_minus_mean.T)
			denominator = alpha[k]-p + sumBETA[k]
			updatedCov[k,...] = numerator/denominator
		if adptCov:
			gmm['cov'] = updatedCov
			for k in range(K):
				gmm['prec'][k] = inv(gmm['cov'][k])

	# ========== Update prior hyper-parameters
	#		We don't need to update v and tau, since these depend on priorRatio
	#   alpha: (K,1)
	#   mu: (p,K)
	#   sigma: (K,p,p)
	# 	sumBETA: (K,1)
	# 	xbar: (p,K)
	rtn_hyper = {}
	rtn_hyper['alpha'] = alpha + sumBETA
	numerator = tau.T*mu + sumBETA.T*xbar # (p,K)
	denominator = tau + sumBETA	# (K,1)
	rtn_hyper['mu'] = numerator/denominator.T # (p,K)
	rtn_hyper['sigma'] = np.zeros([K,p,p])
	mu_minus_xbar = mu-xbar # (p,K)
	factor = (tau*sumBETA)/(tau+sumBETA) # (K,1)
	for k in range(K):
		rtn_hyper['sigma'][k,...] = sigma[k,...] + S[k,...] + factor[k]*np.matmul(mu_minus_xbar[:,k].reshape([-1,1]),mu_minus_xbar[:,k].reshape([1,-1]))

	return gmm, rtn_hyper


if __name__ == '__main__':
	# 2D Toy Example
	gmm = {'mu':[[-4, -6],[4, -3],[0, 3]],\
				'cov':[[[1, 3.0/5.0], [3.0/5.0, 2]], [[3.0, -1.0], [-1.0, 1.0]], [[2, 3.0/5.0], [3.0/5.0, 1]]],\
				'w':[0.5, 0.25, 0.25],
				'K':3}
	gmm['prec'] = np.zeros([gmm['K'],2,2])
	for k in range(gmm['K']):
		gmm['prec'][k] = inv(gmm['cov'][k])

	N = 100
	X1 = sampleByGaussian([-5, -7], [[3.0, -1.0], [-1.0, 1.0]], N)
	# We let X2 (corresponding to mixture 2's adaptation data) be empty.
	# We do this for checking the behavior of MAP while missing some data for an mixture
	X3 = sampleByGaussian([-1.0, 4.3], [[1, 3.0/5.0], [3.0/5.0, 2]], N/2)
	X = np.concatenate([X1,X3],axis=1)

	# Distributions for UBM
	pts1 = guassianPlot2D(gmm['mu'][0], gmm['cov'][0])
	pts2 = guassianPlot2D(gmm['mu'][1], gmm['cov'][1])
	pts3 = guassianPlot2D(gmm['mu'][2], gmm['cov'][2])

	# Do MAP 10 times to simulate the on-line adaptation for data is coming periodically
	priorRatio = 0.9330329915368074	# similar to only using ONE MAP with priorRatio = 0.5. i.e. gmmMap(gmm,X,0.5,adptW=False,adptM=True,adptCov=False)
	# priorRatio = 0.7943282347242815	# similar to only using ONE MAP with priorRatio = 0.1. i.e. gmmMap(gmm,X,0.1,adptW=False,adptM=True,adptCov=False)
	gmm, rtn_hyper = gmmMap(gmm,X,priorRatio,adptW=False,adptM=True,adptCov=False)
	for i in range(9):
		gmm, rtn_hyper = gmmMap(gmm,X,priorRatio,adptW=False,adptM=True,adptCov=False,hyper=rtn_hyper)

	# Distributions after MAP
	pts1_map = guassianPlot2D(gmm['mu'][0], gmm['cov'][0])
	pts2_map = guassianPlot2D(gmm['mu'][1], gmm['cov'][1])
	pts3_map = guassianPlot2D(gmm['mu'][2], gmm['cov'][2])

	# Plotting area
	plt.figure()
	plt.scatter(X[0],X[1], c='', edgecolors='b', marker='o')
	plt.plot(pts1_map[0,:],pts1_map[1,:],'r-')
	plt.plot(pts1[0,:],pts1[1,:],'g--')
	plt.legend(['After MAP','UBM','Adpt data'])

	plt.plot(pts2_map[0,:],pts2_map[1,:],'r-')
	plt.plot(pts2[0,:],pts2[1,:],'g--')

	plt.plot(pts3_map[0,:],pts3_map[1,:],'r-')
	plt.plot(pts3[0,:],pts3[1,:],'g--')

	plt.axis('equal')
	plt.xlim([-12,12])
	plt.ylim([-11,7])
	plt.show()