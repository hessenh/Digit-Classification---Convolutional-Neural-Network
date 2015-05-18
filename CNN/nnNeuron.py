from nnConnection import NNConnection
class NNNeuron:

	output = 0
	connections = []
	layerNumber = None

	def __init__(self):
		self.output = 0
		self.connections = []
		self.layerNumber = None


	def addConnection(self,neuronIndex,iNumWeights):
		c = NNConnection(neuronIndex,iNumWeights)
		self.connections.append(c)
