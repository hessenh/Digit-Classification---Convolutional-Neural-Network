import numpy as np, scipy.signal as ss,loadData

def convertToNewFormatTarget(iList):
	nList = []

	for t in iList:
		l = []
		for i in range(0,10):
			if i==t:
				l.append(1.0)
			else:
				l.append(0.0);
		nList.append(l);
	return nList


class BackPropegationNetwork:

	#
	# Class members
	#
	layerCount = 0 
	convolutionCount = 0
	subSamplingCount = 0
	convolutionShape = None
	shape = None
	weights = []

	#
	# Class methods
	#
	def __init__(self,convolutionSize,layerSize):
		#Intialize the network - layerSize ect (2,3,2)
		
		# Layer info
		self.layerCount = len(layerSize)-1
		self.shape = layerSize

		#Covolution info
		self.convolutionCount = len(convolutionSize)/2
		self.convolutionShape = convolutionSize

		# Subsampling 
		self.subSamplingCount = len(convolutionSize)/2

		# Input/Output data form the last run
		self._layerInput = []
		self._layerOutput = []

		# Add the convolution part
		for k in range(self.convolutionCount):
			# Add random kernel with size 
			self.weights.append(np.random.normal(scale=0.1,size=(self.convolutionShape[k*2],self.convolutionShape[k*2])))
			#self.weights.append(np.ones((self.convolutionShape[k*2],self.convolutionShape[k*2])))
		
			# Add the subsampling. With ones = average.
			self.weights.append(np.ones((self.convolutionShape[k*2+1],self.convolutionShape[k*2+1])))
		
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
		for index in range(self.convolutionCount*2 + self.layerCount):
			#Input layer - convolve 
			if index ==0:
				layerInput = ss.convolve2d(input,self.weights[0], mode='same')
				#print input,len(input)
				#print self.weights[0],len(self.weights[0])
				#print layerInput,len(layerInput)

			# Convolution layers - Alter between convolute and subsampling. Start with averge	
			elif index>0 and index<self.convolutionCount*2:
				layerInput = ss.convolve2d(self._layerOutput[-1],self.weights[index], mode='same')
				#print self._layerOutput[-1],len(self._layerOutput[-1])
				#print self.weights[index],len(self.weights[index])
				#print layerInput,len(layerInput)
			
			# First layer of Neural network
			elif index ==self.convolutionCount*2:
				#print self._layerOutput[-1],len(self._layerOutput[-1])
				#print self.weights[index],len(self.weights[index])
				#print layerInput,len(layerInput)
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1].T,np.ones([1,lnCases])]))

			# Rest of neural network
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
		for index in reversed(range(self.layerCount+self.convolutionCount*2)):
			if index == self.layerCount+self.convolutionCount*2-1:
				# Compare to the target values to get delta
				print self._layerOutput[index],len(self._layerOutput[index])
				print target.T
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
	#First input of NN have to be the same as output of conv...!!
	bpn = BackPropegationNetwork((4,4,3,3),(28,10,1))
	print bpn.convolutionShape,bpn.shape
	
	images,imagesTarget = loadData.getImages(100)
	imagesTarget = convertToNewFormatTarget(imagesTarget)

	lnMax = 100000
	lnErr = 1e-5

	lvInput = np.array([[ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.]])

	print bpn.Run(np.array(images[0]))
	err = bpn.TrainEpoch(np.array(images[0]),np.array([imagesTarget[0]]))
	# for i in range(lnMax-1):
	# 	err = bpn.TrainEpoch(lvInput,lvTarget)
	# 	if i%2500 == 0:
	# 		print "Iteration {0}\tError: {1:0.6f}".format(i,err)
	# 	if err <= lnErr:
	# 		print "Minimum error reached at iteration {0}".format(i)
	# 		break
	# # Display output
	# lvOutput = bpn.Run(lvInput)
	# print "Input : {0}\nOutput:{1}".format(lvInput,lvOutput)
