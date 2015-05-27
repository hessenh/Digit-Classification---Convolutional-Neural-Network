from layer import Layer

class ConvolutionalNeuralNetwork:

	layers = []
	learningRate = 0.001
	outputVector = []

	#Constructor of ConvolutionalNeuralNetwork, it sets the learning rate from the parameter
	def __init__(self,learningRate):
		self.learningRate = learningRate

	#
	# Adds a layer into the network
	#
	def addLayer(self,layer):
		self.layers.append(layer)

	#
	# ForwardPasss the forwardpass in the network
	#
	def ForwardPass(self,inputVector):
		#
		# Forwardpass - iterate over every layer
		#
		for l in range(len(self.layers)):
			#
			# First layer is input, set outputs of neurons to the input vector 	
			#

			if(l==0):
				
				# Iterate over every Neuron in that layer
				for n in range(len(self.layers[l].neurons)):

					# Set the output of that neuron to the input
					self.layers[l].neurons[n].output = inputVector[n]

			else:
				
				# Iterate over every neuron and call ForwardPass function
				#for n in range(len(self.layers[l].neurons)):
				self.layers[l].ForwardPass()


		#
		# Set the output vector - Iterate over last layer and get output from each neuron
		#		
		lastLayer = len(self.layers[-1].neurons)
		self.outputVector = []
		for i in range(0,lastLayer):
			self.outputVector.append(self.layers[len(self.layers)-1].neurons[i].output)





	#
	# Calculate the backpropagation of the neural network
	#
	def Backpropagate(self,actualOutput,desiredOutput):

		dErr_wrt_dXlast = []
		differentials = []

		#
		# Calculate the difference between target and actual output
		#
		for i in range(0,len(self.layers[-1].neurons)):

			dErr_wrt_dXlast.append(actualOutput[i] - desiredOutput[i])



		iSize = len(self.layers)

		for i in range(0,iSize):
			differentials.append([])
		

		differentials[iSize-1] = dErr_wrt_dXlast

		#
		# Iterate through all but the first layer and run backpropagate on each layer
		#
		for i in range(iSize-1,1,-1):

			# Second input is pointer to return value...
			if self.learningRate <0:
				self.learningRate = 0.0000001

			differentials[i-1] = self.layers[i].Backpropagate(differentials[i],differentials[i-1],self.learningRate)
			#self.layers[i].Backpropagate(differentials[i],differentials[i-1],self.learningRate)


		differentials = []
