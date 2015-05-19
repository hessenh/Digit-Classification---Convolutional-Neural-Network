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
		#
		#print "Calculating forward pass on layerNumber:",self.layerNumber

		# Iterate over all neurons 
		for n in range(0,len(self.neurons)):

			s = self.weights[self.neurons[n].connections[0].weightIndex].val

			# Iterate over all connections 
			for c in range(0,len(self.neurons[n].connections)):
				#
				# Just divided them for better understanding
				one = self.weights[self.neurons[n].connections[c].weightIndex].val
				two = self.prevLayer.neurons[self.neurons[n].connections[c].neuronIndex].output
				s += one*two

			# Update the neurons new output using sigmoid function
			self.neurons[n].output = math.tanh(s)


	def Backpropagate(self,dErr_wrt_dXn,dErr_wrt_dXnm1,learningRate):
		print "NNLayer back"
		#
		# Calculate (3) : dErr_wrt_dYn = F'(Yn) * dErr_wrt_Xn
		#
		dErr_wrt_dYn = []

		for i in range(0,len(self.neurons)):

			# Get output from neuron
			output = self.neurons[i].output

			dErr_wrt_dYn.append((1.0-math.tanh(output)**2.0)*1.0* dErr_wrt_dXn[i])


		#
		# Calculate (4) : dErr_wrt_Wn = Xnm1 * dErr_wrt_Yn
		# For each neuraon in the layer go through the list of connections from the prior layer,
		# and upate the differential for the corresponding weight
		#
		dErr_wrt_dWn = []
		##### 
		#### This array is too big! Have no idea how long it should be
		####
		dErr_wrt_dWn.extend(range(0, 1000000))
	
		i = 0
		# Over all neurons 
		for n in self.neurons:

			for c in n.connections:

				if c.neuronIndex==-1: #Bias node
					output = 1.0

				else:
					output = self.prevLayer.neurons[c.neuronIndex].output

				#print c.weightIndex,len(dErr_wrt_dWn)
				dErr_wrt_dWn[c.weightIndex] += dErr_wrt_dYn[i]*1.0 *output
				


			i+=1


		#
		# Calculate (5) : dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn
		# which is needed as the input value of
   		# dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
		#

		##### 
		#### This array is too big! Have no idea how long it should be
		####
		dErr_wrt_dXnm1 = []
		for i in range(0,10000):
			dErr_wrt_dXnm1.append(0)
		
		i = 0
		# Over each neuron
		for n in self.neurons:

			for c in n.connections:

				k = c.neuronIndex

				if k !=0:
					temp =  dErr_wrt_dXnm1[k]
					dErr_wrt_dXnm1[k] = temp + dErr_wrt_dYn[i]*1.0 * self.weights[c.weightIndex].val

			i+=1


		#
		# Calculate (6) : Upate the weights 
		# 
		for i in range(len(self.weights)):
			oldValue = self.weights[i].val
			newValue = oldValue - learningRate*1.0*dErr_wrt_dWn[i]
			self.weights[i].val = newValue

		return dErr_wrt_dXnm1
