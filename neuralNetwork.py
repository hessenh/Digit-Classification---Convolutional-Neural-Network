import numpy as np


class BackPropegationNetwork:

	#
	# Class members
	#
	layerCount = 0 
	shape = None
	weights = []

	#
	# Class methods
	#
	def __init__(self,layerSize):
		#Intialize the network - layerSize ect (2,3,2)
		
		# Layer info
		self.layerCount = len(layerSize)-1
		self.shape = layerSize

		# Input/Output data form the last run
		self._layerInput = []
		self._layerOutput = []

		# Create the weight arrays
		for (l1,l2) in zip(layerSize[:-1],layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1,size = (l2,l1+1))) 


	#
	#	Run method
	#
	def Run(self,input):
		"""Run the network based on the input data"""
		lnCases = input.shape[0]

		# Clear out previous intermediate value list 
		self._layerInput = []
		self._layerOutput = []

		# Run it!
		for index in range(self.layerCount):
			#Determine layer input
			if index == 0:
				layerInput = self.weights[0].dot(np.vstack([input.T,np.ones([1,lnCases])]))
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,lnCases])]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))


		return self._layerOutput[-1].T 

	#
	#	TrainEpoch method
	#
	def TrainEpoch(self,input, target,traingRate = 0.1):
		"""This mehtod trains the network for one epoch"""

		delta = [] 
		lnCases = input.shape[0]

		# First - run the network
		self.Run(input)

		# Calculate deltas
		for index in reversed(range(self.layerCount)):
			if index == self.layerCount-1:
				# Compare to the target values to get delta
				output_delta = self._layerOutput[index]-target.T
				error = np.sum(output_delta**2)
				delta.append(output_delta*self.sgm(self._layerInput[index],True))
			else:
				# Compate to the following layers delta
				delta_pullback = self.weights[index+1].T.dot(delta[-1])
				delta.append(delta_pullback[:-1,:] * self.sgm(self._layerInput[index],True))

		# Compute weight deltas 
		for index in range(self.layerCount):
			delta_index = self.layerCount-1 -index

			if index == 0:
				layerOutput = np.vstack([input.T,np.ones([1,lnCases])])
			else:
				layerOutput = np.vstack([self._layerOutput[index-1],np.ones([1,self._layerOutput[index-1].shape[1]])])

			weightDelta = np.sum(\
								layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0)\
								,axis=0)
			self.weights[index] -= traingRate * weightDelta

		return error


	# Transfer functions
	def sgm(self,x,Derivative=False):
		if not Derivative:
			return 1 / (1+np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)

#
# If run as a script, create a test object
#

if __name__ == "__main__":
	bpn = BackPropegationNetwork((2,2,2))
	print bpn.shape
	print bpn.weights

	lvInput = np.array([[0,0],[1,1],[0,1],[1,0]])
	lvTarget =np.array([[0.05],[0.05],[0.95],[0.95]])

	lnMax = 100000
	lnErr = 1e-5

	for i in range(lnMax-1):
		err = bpn.TrainEpoch(lvInput,lvTarget)
		if i%2500 == 0:
			print "Iteration {0}\tError: {1:0.6f}".format(i,err)
		if err <= lnErr:
			print "Minimum error reached at iteration {0}".format(i)
			break
	# Display output
	lvOutput = bpn.Run(lvInput)
	print "Input : {0}\nOutput:{1}".format(lvInput,lvOutput)
