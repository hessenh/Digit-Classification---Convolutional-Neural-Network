import loadData,cnnConvolve,generateFilters,cnnPool as pool
from matplotlib import pyplot as plt, numpy as np
from neuralNetwork import *


kernels = generateFilters.getKernels(1,5)

images,imagesTarget = loadData.getImages(100)
testImagesData,testImagesTarget = loadData.getImages(10);

#Display the  images
for i in range(0,0):
	imgplot = plt.imshow(images[i])
	plt.show()

#Convolve images with filters
cImages = cnnConvolve.convolveImages(images,kernels)
TestCImages = cnnConvolve.convolveImages(testImagesData,kernels)


#Display the conved images
for i in range(0,0):
	imgplot = plt.imshow(cImages[i])
	plt.show()

#Average pooling
pooledFeauters = pool.averagePool(10,cImages)
testPooledFeauters = pool.averagePool(10,TestCImages)


#Dispaly pooling
for i in range(0,0):
	imgplot = plt.imshow(pooledFeauters[i])
	plt.show()



def convertToNewFormatInput(iList):
	nList = []

	for img in iList:
		l = []
		for r in img:
			for p in r:
				l.append(p)
		nList.append(l)
	return nList

def convertToNewFormatTarget(iList):
	nList = []

	for t in iList:
		l = []
		for i in range(0,10):
			if i==t:
				l.append(1.0)
			else:
				l.append(0.0);
		nList.append(l);
	return nList


def train_network(trainData,trainTarget,testData):
	bpn = BackPropegationNetwork((2,4,1))
	#bpn = BackPropegationNetwork((len(trainData[0]),len(trainData[0]),15,12,len(trainTarget[0])))
	print bpn.shape
	print bpn.weights

	#lvInput = np.array(trainData)
	#lvTarget = np.array(trainTarget)
	#lvTest = np.array(testData)
	lvInput = np.array([[0,0],[1,1],[0,1],[1,0]])
	lvTarget =np.array([[0.05],[0.05],[0.95],[0.95]])

	lnMax = 100000
	lnErr = 1e-5

	for i in range(lnMax-1):
		err = bpn.TrainEpoch(lvInput,lvTarget)
		if i%10000 == 0:
			print "Iteration {0}\tError: {1:0.6f}".format(i,err)
		if err <= lnErr:
			print "Minimum error reached at iteration {0}".format(i)
			break
	# Display output
	lvOutput = bpn.Run(lvInput)
	print "Input : {0}\nOutput:{1}".format(lvInput,lvOutput)
	print " " 
	print bpn.weights[0]

# This is not a good way to change the format...
inputList = convertToNewFormatInput(pooledFeauters)
imagesTarget = convertToNewFormatTarget(imagesTarget)

testImagesData = convertToNewFormatInput(testPooledFeauters)
testImagesTarget = convertToNewFormatTarget(testImagesTarget)
#Get stucked..
train_network(inputList,imagesTarget,testImagesData)


