from neuralNetwork import NeuralNetwork
from nnLayer import NNLayer
import random
import loadData


if __name__ == "__main__":


	# Initialize and build neural network
	nn = NeuralNetwork()


	#
	# Layer zero, the input layer
	# Create neurons: the number of neurons is the same as the input
	# List of 29*29=841 pixels, and no weights/connections
	#
	layer0 = NNLayer("layer0")

	for i in range(0,841):
		layer0.addNeuron()
	
	nn.addLayer(layer0)

	#
	# Layer 1: Convolutional layer
	# 6 feature maps. Each feature map is 13x13, and each unit in the feature map is a 5x5 convolutional kernel 
	# from the input layer.
	# So there are 13x13x6 = 1014 neurons, (5x5+1)x6 wights
	#
	layer1 = NNLayer("layer1")
	layer1.setPrevLayer(layer0)


	# Add the neurons
	for i in range(0,1014):
		layer1.addNeuron()

	# Add wights
	for i in range(0,156):
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
	for fm in range(0,6):

		for i in range(0,13):

			for j in range(0,13):

				# 26 is the number of weights per featuremaps
				iNumWeights = fm * 26;

				# Bias weight
				layer1.neurons[fm*169+j+i*13].addConnection(iNumWeights,0)
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


	layer2 = NNLayer("layer2")
	layer2.setPrevLayer(layer1)

	# Add the neurons
	for i in range(0,1250):
		layer2.addNeuron()

	# Add wights
	for i in range(0,7800):
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


	for fm in range(0,50):

		for i in range(0,5):

			for j in range(0,5):

				 # 26 is the number of weights per featuremaps
				iNumWeight = fm * 26;

				# Bias weight
				layer2.neurons[fm*25+j+i*5].addConnection(iNumWeight,0)
				iNumWeight +=1

				for k in range(0,25):

					layer2.neurons[fm*25+j+i*5].addConnection(	    2*j+26*i+kernelTemplate[k],iNumWeight)
					iNumWeight +=1
					layer2.neurons[fm*25+j+i*5].addConnection(169 + 2*j+58*i+kernelTemplate[k],iNumWeight)
					iNumWeight +=1
					layer2.neurons[fm*25+j+i*5].addConnection(338 + 2*j+58*i+kernelTemplate[k],iNumWeight)
					iNumWeight +=1
					layer2.neurons[fm*25+j+i*5].addConnection(507 + 2*j+58*i+kernelTemplate[k],iNumWeight)
					iNumWeight +=1
					layer2.neurons[fm*25+j+i*5].addConnection(676 + 2*j+58*i+kernelTemplate[k],iNumWeight)
					iNumWeight +=1
					layer2.neurons[fm*25+j+i*5].addConnection(845 + 2*j+58*i+kernelTemplate[k],iNumWeight)
					iNumWeight +=1

	
	# add layer to network
	nn.addLayer(layer2)



	#
	# layer three:
    # This layer is a fully-connected layer
    # with 100 units.  Since it is fully-connected,
    # each of the 100 neurons in the
    # layer is connected to all 1250 neurons in
    # the previous layer.
    # So, there are 100 neurons and 100*(1250+1)=125100 weights
	#
	
	layer3 = NNLayer("layer3")
	layer3.setPrevLayer(layer2)

	# Add the neurons
	for i in range(0,100):
		layer3.addNeuron()

	# Add wights
	for i in range(0,125100):
		# Uniform random distribution
		initWeight = 0.05*random.uniform(-1,1)

		layer3.addWeight(initWeight)

	# Interconnections with previous layer: fully-connected

	iNumWeight = 0 # Weights are not shared in this layer

	for fm in range(0,100):
		layer3.neurons[fm].addConnection(iNumWeight,0) #bias
		iNumWeight+=1

		for i in range(0,1250):

			layer3.neurons[fm].addConnection(iNumWeight,i) #bias
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

	layer4 = NNLayer("layer4")

	layer4.setPrevLayer(layer3)


	# Add the neurons
	for i in range(0,10):
		layer4.addNeuron()

	# Add wights
	for i in range(0,1010):
		# Uniform random distribution
		initWeight = 0.05*random.uniform(-1,1)

		layer4.addWeight(initWeight)

	# Interconnections with previous layer: fully-connected

	iNumWeight = 0 # Weights are not shared in this layer

	for fm in range(0,10):

		layer4.neurons[fm].addConnection(iNumWeight,0) #bias
		iNumWeight+=1

		for i in range(0,100):

			layer4.neurons[fm].addConnection(iNumWeight,i) #bias
			iNumWeight+=1

	# Add layer to network
	nn.addLayer(layer4)	



	print "Layer 0:",len(nn.layers[0].neurons)
	print "Layer 1:",len(nn.layers[1].neurons)
	print "Layer 2:",len(nn.layers[2].neurons)
	print "Layer 3:",len(nn.layers[3].neurons)
	print "Layer 4:",len(nn.layers[4].neurons)

	print len(nn.layers[0].neurons[0].connections)
	print len(nn.layers[1].neurons[0].connections)
	print len(nn.layers[2].neurons[0].connections)
	# THis is not a good thing. Have to find a way to padd the picture
	i,t = loadData.getRandomImage()
	nn.Calculate(i)
	print nn.outputVector
	print len(i)