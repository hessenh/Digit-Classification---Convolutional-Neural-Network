from nnLayer import NNLayer

class NeuralNetwork:

	layers = []

	outputVector = []

	def __init__(self):

		print "self"

	def addLayer(self,layer):
		self.layers.append(layer)

	def Calculate(self,inputVector):
		print "calc"

		#
		# Forwardpass - iterate over every layer
		#
		for l in range(len(self.layers)):
			print self.layers[l].layerNumber
			#
			# First layer is input, set outputs of neurons to the input vector 	
			#

			if(l==0):
				
				# Iterate over every Neuron in that layer
				for n in range(len(self.layers[l].neurons)):

					# Set the output of that neuron to the input
					self.layers[l].neurons[n].output = inputVector[n]

			else:
				
				# Iterate over every neuron and call calculate function
				#for n in range(len(self.layers[l].neurons)):
				self.layers[l].Calculate()


		#
		# Set the output vector - Iterate over last layer and get output from each neuron
		#		
		lastLayer = len(self.layers[-1].neurons)
		for i in range(0,lastLayer):
			self.outputVector.append(self.layers[len(self.layers)-1].neurons[i].output)



	def Backpropagate(actualOutput,desiredOutput,count):
		print "back"


		dErr_wrt_dXlast = []
		differentials = []

		#
		# Calculate the difference between target and actual output
		#
		for i in range(0,len(layers[-1].neurons)):
			dErr_wrt_dXlast[i] = actualOutput[i] - desiredOutput[i]



		#
		# Todo ?
		#
		differentials.append(dErr_wrt_dXlast)

		#
		# Iterate through all but the first layer and run backpropagate on each layer
		#
		for i in reversed(len(layers),1,-1):
			## TODO
			# Second input is pointer to return value...
			layers[i].Backpropagate(differentials[i],differentials[i-1],learningRate)


