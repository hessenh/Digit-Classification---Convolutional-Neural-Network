import random,numpy as np

def randomKernel(kernelSize):
	kernel = []
	for i in range(kernelSize):
		t = []
		for j in range(kernelSize):
			r  = random.randrange(-1000,1000)*1.0/1000
			t.append(r)
		kernel.append(t)
	return kernel

def getKernels(nKernels,kernelSize):
	filters = []
	for i in range(nKernels):
		filters.append(randomKernel(kernelSize))
	return filters

def getAverageFilter(dim):
	return [[1]*dim]*dim
