from nnWeight import NNWeight
from neuron import Neuron
import math

class Layer:

	connections = []
	neurons = []
	weights = []
	prevLayer = None
	layerNumber = None
	numNeurons = 0

	# Initializes a layer and gives the layerNumber from the parameter in its constructor
	def __init__(self,layerNumber,numNeurons):
		self.numNeurons = numNeurons
		self.layerNumber = layerNumber
		self.neurons = []
		self.connections = []
		self.weights = []

	# Adds a neorun into the list of neurons in the layer
	def addNeuron(self):
		n = Neuron()
		self.neurons.append(n)

	# Sets the previous layer
	def setPrevLayer(self,layer):
		self.prevLayer = layer

	# sets the weight list with its values
	def loadWeights(self,weights):
		self.weights = weights

	# Adds a weight into the weight list
	def addWeight(self,initWeight):
		w = NNWeight(initWeight)
		self.weights.append(w)


	# ForwardPasss the forward pass in the layer
	def ForwardPass(self):
		#
		#print "Calculating forward pass on layerNumber:",self.layerNumber

		# Iterate over all neurons 
		for n in range(0,len(self.neurons)):

			s = self.weights[self.neurons[n].connections[0].weightIndex].val

			# Iterate over all connections 
			for c in range(1,len(self.neurons[n].connections)):
				#
				# Just divided them for better understanding
				one = self.weights[self.neurons[n].connections[c].weightIndex].val
				two = self.prevLayer.neurons[self.neurons[n].connections[c].neuronIndex].output
				s += one*two

			# Update the neurons new output using sigmoid function
			self.neurons[n].output = (1.7159*math.tanh((2.0/3.0)*s));#math.tanh(s)


	# Calc the backpropagation for that layer
	def Backpropagate(self,dErr_wrt_dXn,dErr_wrt_dXnm1,learningRate):

		#
		# Calc (3) : dErr_wrt_dYn = F'(Yn) * dErr_wrt_Xn
		#
		dErr_wrt_dYn = []

		for i in range(0,len(self.neurons)):

			# Get output from neuron
			output = self.neurons[i].output


			#
			# Is this correct? 
			# 
			t = ((2.0/3.0)/1.7159*(1.7159+(output))*(1.7159-(output)))
			dErr_wrt_dYn.append(t* dErr_wrt_dXn[i])

			#dErr_wrt_dYn.append((1.0-math.tanh(output))* dErr_wrt_dXn[i])


		#
		# Calc (4) : dErr_wrt_Wn = Xnm1 * dErr_wrt_Yn
		# For each neuron in the layer go through the list of connections from the prior layer,
		# and upate the differential for the corresponding weight
		#
		dErr_wrt_dWn = []
		##### 
		#### This array is too big! Have no idea how long it should be
		####
		for i in range(0,len(self.weights)):
			dErr_wrt_dWn.append(0)
	
		i = 0
		# Over all neurons 
		for n in self.neurons:

			for c in n.connections:

				if c.neuronIndex==-10000: #Bias node
					output = 1.0

				else:
					output = self.prevLayer.neurons[c.neuronIndex].output

				#print c.weightIndex,len(dErr_wrt_dWn)
				dErr_wrt_dWn[c.weightIndex] += dErr_wrt_dYn[i]*output
				


			i+=1


		#
		# Calc (5) : dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn
		# which is needed as the input value of
   		# dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
		#

		##### 
		#### This array is too big! Have no idea how long it should be
		####
		dErr_wrt_dXnm1 = []
		for i in range(0,1250):
			dErr_wrt_dXnm1.append(0)
		
		i = 0
		# Over each neuron
		for n in self.neurons:

			for c in n.connections:

				k = c.neuronIndex

				if k !=-10000:
					temp =  dErr_wrt_dXnm1[k]
					dErr_wrt_dXnm1[k] += dErr_wrt_dYn[i]*self.weights[c.weightIndex].val

			i+=1


		#
		# Calculate (6) : Upate the weights 
		# 
		for i in range(len(self.weights)):
			oldValue = self.weights[i].val
			newValue = oldValue - learningRate*dErr_wrt_dWn[i]
			self.weights[i].val = newValue

		return dErr_wrt_dXnm1
