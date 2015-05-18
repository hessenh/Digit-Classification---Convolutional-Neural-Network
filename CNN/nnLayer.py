from nnWeight import NNWeight
from nnNeuron import NNNeuron
import math

class NNLayer:

	connections = []
	neurons = []
	weights = []
	prevLayer = None
	layerNumber = None

	def __init__(self,layerNumber):
		self.layerNumber = layerNumber
		self.neurons = []
		self.connections = []
		self.weights = []

	def addNeuron(self):
		n = NNNeuron()
		self.neurons.append(n)


	def setPrevLayer(self,layer):
		self.prevLayer = layer


	def addWeight(self,initWeight):
		w = NNWeight(initWeight)
		self.weights.append(w)



	def Calculate(self):
		#print "NNLayer calc"
		#
		print "Calculating forward pass on layerNumber",self.layerNumber

		# Iterate over all neurons 
		for n in range(0,len(self.neurons)):

			s = self.weights[self.neurons[n].connections[0].weightIndex].val

			#print self.neurons[n].connections[0].weightIndex,"<",len(self.weights)
			#
			# Iterate over all connections 
			for c in range(0,len(self.neurons[n].connections)):
				#
				#
				one = self.weights[self.neurons[n].connections[c].weightIndex].val

				#print len(self.prevLayer.neurons),self.neurons[n].connections[c].neuronIndex
				#print c,len(self.neurons[n].connections)
				two = self.prevLayer.neurons[self.neurons[n].connections[c].neuronIndex].output
				s += one*two
				#print s,c
			# Update the neurons new output using sigmoid function
			self.neurons[n].output = math.tanh(s)


	def Backpropagate(dErr_wrt_dXn,dErr_wrt_dXnm1,learningRate):
		print "NNLayer back"
		#
		# Calculate (3) : dErr_wrt_dYn = F'(Yn) * dErr_wrt_Xn
		#
		dErr_wrt_dYn = []

		for i in range(0,len(neurons)):

			# Get output from neuron
			output = neurons[i].output

			dErr_wrt_dYn[i] = DSigmoid(output) * dErr_wrt_dXn[i]


		#
		# Calculate (4) : dErr_wrt_Wn = Xnm1 * dErr_wrt_Yn
		# For each neuraon in the layer go through the list of connections from the prior layer,
		# and upate the differential for the corresponding weight
		#
		i = 0
		# Over all neurons 
		for n in neurons:

			for c in connections:

				if c.neuronIndex==-1: #Bias node
					output = 1.0

				else:
					output = prevLayer.neurons[c.neuronIndex].output

				dErr_wrt_dWn[c.weightIndex] += dErr_wrt_dYn[i] *output
			
			i+=1


		#
		# Calculate (5) : dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn
		# which is needed as the input value of
   		# dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
		#

		i = 0
		# Over each neuron
		for n in neurons:

			for c in connections:

				k = c.neuronIndex
				if k !=-1:

					dErr_wrt_dXnm1[k] += dErr_wrt_dYn[i] * n.weights[c.weightIndex]

			i+=1


		#
		# Calculate (6) : Upate the weights 
		# 
		for i in range(len(weights)):
			oldValue = weights[i]
			newValue = oldValue - learningRate*dErr_wrt_dWn[i]
			weight[i] = newValue

		return dErr_wrt_dXnm1
