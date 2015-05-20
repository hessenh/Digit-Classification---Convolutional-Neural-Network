import scipy.signal as ss,numpy as np,generateFilters as gs


def averagePool(poolDim, convolvedFeatures):
	kernel = gs.getAverageFilter(poolDim)
	result = []
	for feature in convolvedFeatures:
		result.append(ss.convolve2d(feature, kernel, mode='valid'))
	return result

	
