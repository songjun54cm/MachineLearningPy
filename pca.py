import scipy as sp
from scipy import linalg
def pca(x, k):
	# x <n x d>, n sample with d dimension
	# k the top k principle component
	num = x.shape[0]
	dim = x.shape[1]
	feaMean = sp.mean(x, axis=0)
	x = x - feaMean
	# if dim>>num, transpose x before eigen calculate and transform back later
	if dim > 2*num:
		new_x = x.T
	else:
		new_x = x
	cov_matrix = sp.dot(new_x.T, new_x)/num
	eig_val, eig_vec = linalg.eig(cov_matrix)
	eig_vec = eig_vec[:,0:k]
	# if dim>>num, transform back
	if dim>2*num:
		eig_vec = x.T * eig_vec
		eig_norm = linalg.norm(eig_vec,axis=0)
		eig_vec = eig_vec/eig_norm
	return eig_vec