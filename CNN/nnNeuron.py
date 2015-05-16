from nnConnection import NNConnection
class NNNeuron:

	output = 0
	connections = []
	layerNumber = None


	def addConnection(self,neuronIndex,iNumWeights):
		self.connections.append(NNConnection(neuronIndex,iNumWeights))
