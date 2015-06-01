from neuralNetwork import ConvolutionalNeuralNetwork
from layer import Layer
from nnWeight import NNWeight
import random
import loadData,visualise
from saveWeights import saveWeights,loadWeights

import scipy
import base64
import re 
import matplotlib
import numpy as np


def initNetwork(nk1,nk2,nN3):

	numKernelOne = nk1
	numKernelTwo = nk2

	numNeuronsThree = nN3
	numNeuronsFour= 10


	#nN0,nN1,nN2,nN3
	#
	# Initialize neural network
	# The parameter sendt in is the learningRate of ther neural network,
	# in this case we set it to 0.001
	#
	nn = ConvolutionalNeuralNetwork(0.001)

	#
	# Layer 0, the input layer
	#
	numNeuronsZero = 841
	layer0 = Layer("layer0",numNeuronsZero)


	# Creates the neurons in the layer0 and adds them into the layer. 
	for i in range(0,numNeuronsZero):
		layer0.addNeuron()
		
	# Adds the layer into the neural network
	nn.addLayer(layer0)

	#
	# Layer 1: Convolutional layer
	# 6 feature maps. Each feature map is 13x13, and each unit in the feature map is a 5x5 convolutional kernel 
	# from the input layer.
	# So there are 13x13x6 = 1014 neurons, (5x5+1)x6 weights
	#
	numNeuronsOne = 13*13*numKernelOne
	numWeightsOne = (5*5+1)*numKernelOne

	layer1 = Layer("layer1",numNeuronsOne)

	# Sets the previous layer as layer0
	layer1.setPrevLayer(layer0)


	# Add the neurons
	for i in range(0,numNeuronsOne):
		layer1.addNeuron()

	# Add weights from layer0 to layer1
	for i in range(0,numWeightsOne):
		# Uniform random distribution
		initWeight = 0.05*random.uniform(-1,1)

		layer1.addWeight(initWeight)

	


	# interconnections with previous layer: this is difficult
	# The previous layer is a top-down bitmap
	# image that has been padded to size 29x29
	# Each neuron in this layer is connected
	# to a 5x5 kernel in its feature map, which 
	# is also a top-down bitmap of size 13x13. 
	# We move the kernel by TWO pixels, i.e., we
	# skip every other pixel in the input image

	kernelTemplate = [0,1,2,3,4,29,30,31,32,33,58,59,60,61,62,87,88,89,90,91,116,117,118,119,120]

	#Feature maps
	for fm in range(0,numKernelOne):

		for i in range(0,13):

			for j in range(0,13):

				# 26 is the number of weights per featuremaps
				iNumWeights = fm * 26;

				# Bias weight
				layer1.neurons[fm*169+j+i*13].addConnection(-10000,iNumWeights)
				iNumWeights +=1

				for k in range(0,25):

					layer1.neurons[fm*169+j+i*13].addConnection(2*j+58*i+kernelTemplate[k],iNumWeights)
					iNumWeights +=1


	# Add layer to network
	nn.addLayer(layer1)


	#
	# Layer two: This layer is a convolutional layer 
	# 50 feature maps. Each feature map is 5x5, and each unit in the feature maps is a 5x5 convolutional kernel of
	# corresponding areas of all 6 of the previous layers, each of which is a 13x13 feature map. 
	# So, there are 5x5x50 = 1250 neurons, (5X5+1)x6x50 = 7800 weights

	numNeuronsTwo = 5*5*numKernelTwo
	numWeightsTwo = (5*5+1)*numKernelTwo*numKernelOne

	layer2 = Layer("layer2",numNeuronsTwo)


	layer2.setPrevLayer(layer1)

	# Add the neurons
	for i in range(0,numNeuronsTwo):
		layer2.addNeuron()

	# Add weights
	for i in range(0,numWeightsTwo):
		# Uniform random distribution
		initWeight = 0.05*random.uniform(-1,1)

		layer2.addWeight(initWeight)



	# Interconnections with previous layer: this is difficult
    # Each feature map in the previous layer
    # is a top-down bitmap image whose size
    # is 13x13, and there are 6 such feature maps.
    # Each neuron in one 5x5 feature map of this 
    # layer is connected to a 5x5 kernel
    # positioned correspondingly in all 6 parent
    # feature maps, and there are individual
    # weights for the six different 5x5 kernels.  As
    # before, we move the kernel by TWO pixels, i.e., we
    # skip every other pixel in the input image.
    # The result is 50 different 5x5 top-down bitmap
    # feature maps

	kernelTemplate = [0,  1,  2,  3,  4,13, 14, 15, 16, 17, 26, 27, 28, 29, 30,39, 40, 41, 42, 43, 52, 53, 54, 55, 56 ]


	for fm in range(0,numKernelTwo):

		for i in range(0,5):

			for j in range(0,5):

				 # 26 is the number of weights per featuremaps
				iNumWeight = fm * 26;

				# Bias weight
				layer2.neurons[fm*25+j+i*5].addConnection(-10000,iNumWeight)
				iNumWeight +=1

				for k in range(0,25):


					for f in range(0,numKernelOne):
						layer2.neurons[fm*25+j+i*5].addConnection(169*f+ 2*j+26*i+kernelTemplate[k],iNumWeight)
						iNumWeight +=1

					# layer2.neurons[fm*25+j+i*5].addConnection(	    2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1
					# layer2.neurons[fm*25+j+i*5].addConnection(169 + 2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1
					# layer2.neurons[fm*25+j+i*5].addConnection(338 +  2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1
					# layer2.neurons[fm*25+j+i*5].addConnection(507 + 2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1
					# layer2.neurons[fm*25+j+i*5].addConnection(676 +  2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1
					# layer2.neurons[fm*25+j+i*5].addConnection(856 + 2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1
					# layer2.neurons[fm*25+j+i*5].addConnection(1014 +  2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1
					# layer2.neurons[fm*25+j+i*5].addConnection(1183 + 2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1
					# layer2.neurons[fm*25+j+i*5].addConnection(1352 + 2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1
					# layer2.neurons[fm*25+j+i*5].addConnection(1521 + 2*j+26*i+kernelTemplate[k],iNumWeight)
					# iNumWeight +=1


	# add layer to network
	nn.addLayer(layer2)



	"""
	layer three:
    This layer is a fully-connected layer
    with 100 neurons.  Since it is fully-connected,
    each of the 100 neurons in the
    layer is connected to all 1250 neurons in
    the previous layer.
    So, there are 100 neurons and 100*(1250+1)=125100 weights
	"""
	
	numWeightsThree = numNeuronsThree*(numNeuronsTwo+1)
	
	layer3 = Layer("layer3",numNeuronsThree)


	layer3.setPrevLayer(layer2)

	# Add the neurons
	for i in range(0,numNeuronsThree):
		layer3.addNeuron()

	# Add wights
	for i in range(0,numWeightsThree):
		# Uniform random distribution
		initWeight = 0.05*random.uniform(-1,1)

		layer3.addWeight(initWeight)

	# Interconnections with previous layer: fully-connected

	iNumWeight = 0 # Weights are not shared in this layer

	for fm in range(0,numNeuronsThree):
		layer3.neurons[fm].addConnection(-10000,iNumWeight) #bias
		iNumWeight+=1

		for i in range(0,numNeuronsTwo):

			layer3.neurons[fm].addConnection(i,iNumWeight) #bias
			iNumWeight+=1

	# Add layer to network
	nn.addLayer(layer3)	



	# layer four, the final (output) layer:
    # This layer is a fully-connected layer
    # with 10 units.  Since it is fully-connected,
    # each of the 10 neurons in the layer
    # is connected to all 100 neurons in
    # the previous layer.
    # So, there are 10 neurons and 10*(100+1)=1010 weights

	
	numWeightsFour = numNeuronsFour*(numWeightsThree+1)

	layer4 = Layer("layer4",numNeuronsFour)


	layer4.setPrevLayer(layer3)


	# Add the neurons
	for i in range(0,numNeuronsFour):
		layer4.addNeuron()

	# Add wights
	for i in range(0,numWeightsFour):
		# Uniform random distribution
		initWeight = 0.05*random.uniform(-1,1)

		layer4.addWeight(initWeight)

	# Interconnections with previous layer: fully-connected

	iNumWeight = 0 # Weights are not shared in this layer

	for fm in range(0,numNeuronsFour):

		layer4.neurons[fm].addConnection(-10000,iNumWeight) #bias
		iNumWeight+=1

		for i in range(0,numNeuronsThree):

			layer4.neurons[fm].addConnection(i,iNumWeight) #bias
			iNumWeight+=1

	# Add layer to network
	nn.addLayer(layer4)	


	print "NN structure:"
	print "Layer 0:",len(nn.layers[0].neurons)
	print "Layer 1:",len(nn.layers[1].neurons)
	print "Layer 2:",len(nn.layers[2].neurons)
	print "Layer 3:",len(nn.layers[3].neurons)
	print "Layer 4:",len(nn.layers[4].neurons)
	print "\n"
	
	return nn


#
# Sets the different weights in the different layers
#
def setWeights(nn,numberOfSet):
	nn.layers[1].loadWeights(loadWeights("1",str(numberOfSet)))
	nn.layers[2].loadWeights(loadWeights("2",str(numberOfSet)))
	nn.layers[3].loadWeights(loadWeights("3",str(numberOfSet)))
	nn.layers[4].loadWeights(loadWeights("4",str(numberOfSet)))

	return nn

#
# Training the network
#
def traingNetwork(nn,numberOfSet):
	print "Training starting:"

	imageNumberList = loadData.getTrainingImageNumberList(numberOfSet)

	d,t = loadData.getImageAndTarget(imageNumberList[0])

	for i in range(1,len(imageNumberList)):
		#print "Forwardpass"
		
		nn.ForwardPass(d)
		

		if(i%(numberOfSet/10)==0):
			print "Number of iterations:",i
			nn.learningRate -=0.00001
		
		if(i==5000):
			saveWeights("1",str(5000),nn.layers[1].weights)
			saveWeights("2",str(5000),nn.layers[2].weights)
			saveWeights("3",str(5000),nn.layers[3].weights)
			saveWeights("4",str(5000),nn.layers[4].weights)
			print "Weights are saved - 5.000.\n"


		if(i==10000):
			saveWeights("1",str(10000),nn.layers[1].weights)
			saveWeights("2",str(10000),nn.layers[2].weights)
			saveWeights("3",str(10000),nn.layers[3].weights)
			saveWeights("4",str(10000),nn.layers[4].weights)
			print "Weights are saved - 10.000.\n"

		if(i==30000):
			saveWeights("1",str(30000),nn.layers[1].weights)
			saveWeights("2",str(30000),nn.layers[2].weights)
			saveWeights("3",str(30000),nn.layers[3].weights)
			saveWeights("4",str(30000),nn.layers[4].weights)
			print "Weights are saved - 30.000.\n"

		nn.Backpropagate(nn.outputVector,t)


		d,t = loadData.getImageAndTarget(imageNumberList[i])


	saveWeights("1",str(numberOfSet),nn.layers[1].weights)
	saveWeights("2",str(numberOfSet),nn.layers[2].weights)
	saveWeights("3",str(numberOfSet),nn.layers[3].weights)
	saveWeights("4",str(numberOfSet),nn.layers[4].weights)
	print "Training completed. Weights are saved.\n"

	return nn
#
# Tests network
#
def testNetwork(nn,numberOfSet,numberOfTest,modification=False):

	print "Testing starting:"

	# Set weights from file
	nn = setWeights(nn,numberOfSet)
	
	imageNumberList = loadData.getTestImageNumberList(numberOfTest)

	correct = 0
	for i in range(0,len(imageNumberList)):

		# Get random picture
		d,t = loadData.getImageAndTarget(imageNumberList[i])

		# Forward-pass
		nn.ForwardPass(d)
		
		correctGuess = False
		# Check if result is correct
		if(nn.outputVector.index(max(nn.outputVector))==t.index(max(t))):
			correct+=1
			correctGuess = True

		print "CNN:",nn.outputVector.index(max(nn.outputVector)),"Target:",t.index(max(t)),correctGuess


	print "\nNumber of correct:",correct
	print "Number of pictures",numberOfTest
	print "Percentage",(correct*1.0/numberOfTest) * 100


#
# Called by the python server
#
def runCNN(nn,image):

	print image
	#
	# This method takes an image and returns the classified label of that image
	#


	#
	# Converts the image to png and saves it localy
	#
	image = image.replace('data:image/png;base64,','')
	fh = open("imageToSave.png", "wb")
	image = image.decode('base64')
	fh.write(image)
	fh.close()


	#
	# Downsample image to 29x29 pixels and saves it
	#


	import PIL
	from PIL import Image

	basewidth = 29	
	img = Image.open('imageToSave.png')
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
	img.save('imageToSave.png')



	# Load the image
	img = matplotlib.image.imread("imageToSave.png")


	#
	# Imported images is on wrong format. [[0,0,0],[0,0,0],[0,0,0]], have to have it in one list
	#
	c = 0
	newimage= []
	for i in range(len(img)):
		for j in range(len(img[i])):
			c = 0
			for k in range(len(img[i][j])):
				c+= img[i][j][k]
			newimage.append(c)


	#
	# Since the image created digital, the image don't have the "natural" features as a handwritten image have
	# Have to do some stuff to make it more applicable.
	#


	# Input picture is something like this:[[0,0,0]
	#										[0,1,0]
	#										[0,0,0]]
	#
	#
	# Output will be something like this:  [[0,   52,  0]
	#										[102, 250, 69]
	#										[0,   94,  0]]
	#


	#
	# For every pixel with value in, the neighbours on all sides of that pixel will have a random value from 50 to 150. 
	#
	padd = [0]*len(newimage)
	for i in range(29*2,len(newimage)-29*2):
		if(newimage[i]>0.12):
			padd[i-1] = random.randint(50,150)
			padd[i+1] = random.randint(50,150)
			#padd[i-2] = 100
			#padd[i+2] = 100

			padd[i-29] = random.randint(50,150)
			padd[i+29] = random.randint(50,150)
			#padd[i-29*2] = 100
			#padd[i+29*2] = 100


	for i in range(len(newimage)):
		if(newimage[i]>0.12 and newimage[i] != 150):
			padd[i] = random.randint(240,255)

	newimage = padd


	#
	# This part is just to look how the created picture is compared to one from the training/testing set
	#
	d,t = loadData.getImageAndTarget(random.randint(0,60000))


	#
	# Get better printing format for both images
	#
	
	#for i in range(0,29):
	#	a = []
	#	for j in range(0,29):
	#		a.append(d[i*29+j])
	#	print a


	#print t.index(max(t))
	#print " "
	#for i in range(0,29):
	#	a = []
	#	for j in range(0,29):
	#		a.append(newimage[i*29+j])
	#	print a
	
	# Forward-pass
	nn.ForwardPass(newimage)

	#
	# Return dictionary to server
	#
	sort = sorted(nn.outputVector[:])[::-1]
	ranked = {}
	for i in range(0,len(nn.outputVector)):
		ranked[i] = (nn.outputVector[nn.outputVector.index(sort[i])],(nn.outputVector.index(sort[i])))

	return ranked
	#return nn.outputVector.index(max(nn.outputVector))


#
# Return image of output neurons of layer to server
#
def getNetworkImage(nn,number):
	d,t = loadData.getImageAndTarget(number)
	nn.ForwardPass(d)
	
	out = visualise.getNeuronOutputs(nn)

	out = np.array(out[0])

	for i in range(0,len(out)):
		
		for j in range(0,len(out[i])):

			out[i][j] = out[i][j]*100

	s = base64.b64encode(out)
	s = "data:image/png;base64,"+s

	return s



#
# Visualises the network
#
def visualiseNetwork(nn,numberOfSet):
	nn = setWeights(nn,numberOfSet)

	d,t = loadData.getImageAndTarget(random.randint(0,60000))
	nn.ForwardPass(d)

	visualise.visualise(nn)


#
# Gets the network
#
def getNetwork():
	cnn = initNetwork()
	cnn = setWeights(cnn,59999)
	return cnn


nn = initNetwork(10,40,60)

trainingSize = 10000
#traingNetwork(nn,trainingSize)


testNetwork(nn,trainingSize,1000)



