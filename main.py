import loadData,cnnConvolve,generateFilters,cnnPool as pool
from matplotlib import pyplot as plt, numpy as np
from neuralNetwork import *


kernels = generateFilters.getKernels(1,5)
images = loadData.getImages(100)


#Convolve images with filters
cImages = cnnConvolve.convolveImages(images,kernels)


#Display the conved images
for i in range(0,0):
	imgplot = plt.imshow(cImages[i])
	plt.show()

#Average pooling
pooledFeauters = pool.averagePool(10,cImages)

#Dispaly pooling
for i in range(0,0):
	imgplot = plt.imshow(pooledFeauters[i])
	plt.show()

print images[0]



def train_network():
	bpn = BackPropegationNetwork((2,2,2))
	print bpn.shape
	print bpn.weights

	lvInput = np.array([[0,0],[1,1],[0,1],[1,0]])
	lvTarget =np.array([[0.05],[0.05],[0.95],[0.95]])

	lnMax = 100000
	lnErr = 1e-5

	for i in range(lnMax-1):
		err = bpn.TrainEpoch(lvInput,lvTarget)
		if i%5000 == 0:
			print "Iteration {0}\tError: {1:0.6f}".format(i,err)
		if err <= lnErr:
			print "Minimum error reached at iteration {0}".format(i)
			break
	# Display output
	lvOutput = bpn.Run(lvInput)
	print "Input : {0}\nOutput:{1}".format(lvInput,lvOutput)

train_network()
