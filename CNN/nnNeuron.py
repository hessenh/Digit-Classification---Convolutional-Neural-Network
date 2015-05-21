from nnConnection import NNConnection
class NNNeuron:

	output = 0
	connections = []
	layerNumber = None

	# Constructor for the Neuron
	def __init__(self):
		self.output = 0
		self.connections = []
		self.layerNumber = None

	# Adds a connection between another neuron and the weight, and appends it into the connection list
	def addConnection(self,neuronIndex,iNumWeights):
		c = NNConnection(neuronIndex,iNumWeights)
		self.connections.append(c)
