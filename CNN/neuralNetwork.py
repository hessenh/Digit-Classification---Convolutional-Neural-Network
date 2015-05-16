from nnLayer import NNLayer

class NeuralNetwork:

	layers = []

	outputVector = []

	def __init__(self):

		print "self"

	def addLayer(self,layer):
		self.layers.append(layer)

	def Calculate(inputVector,iCount,outputVector,oCount=0):
		print "calc"

		#
		# Forwardpass - iterate over every layer
		#
		for l in range(len(layers)):
			#
			# First layer is input, set outputs of neurons to the input vector 	
			#
			if(l==0):
				
				# Iterate over every Neuron in that layer
				for n in range(len(layers[l])):

					# Set the output of that neuron to the input
					layers[l][n].output = inputVector[n]

			else:

				# Iterate over every neuron and call calculate function
				for n in range(len(layers[l])):

					layers[l][n].Calculate()


		#
		# Set the output vector - Iterate over last layer and get output from each neuron
		#		
		lastLayer = len(layers[len(layers)-1])

		for i in range(0,lastLayer):
			outputVector[i] = layers[lastLayer][i].output



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


