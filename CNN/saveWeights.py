import pickle
from nnWeight import NNWeight
from os import path
import os

#
# Save the weights from the training so you dont have to train the network each time you whould like to run it
# it takes in the layer number, the number of different data you have trained on and its weights
#
def saveWeights(layerNumber,numberOfTrainingData,weights):
	file_path = path.relpath("weights/"+numberOfTrainingData+"."+layerNumber+".cvs")
	outfile = open(file_path, "wb")

	pickle.dump(weights, outfile)

	outfile.close()



#
# Loads the weights from the when trained the network in an earlier run,
# it takes in the layer number and the number of training data you have used. 
#
def loadWeights(layerNumber,numberOfTrainingData):

	file_path = path.relpath("weights/"+numberOfTrainingData+"."+layerNumber+".cvs") 
	try:
		infile = open(file_path,"rb")
	except IOError as e:
		print("({})".format(e))
		print "FILE DOES NOT EXSIST"
	return pickle.load(infile)