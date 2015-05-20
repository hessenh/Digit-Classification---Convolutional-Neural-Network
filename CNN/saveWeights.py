import pickle
from nnWeight import NNWeight
from os import path



def saveWeights(layerNumber,numberOfTrainingData,weights):
	file_path = path.relpath("weights/"+numberOfTrainingData+"."+layerNumber+".cvs")
	outfile = open(file_path, "wb")

	pickle.dump(weights, outfile)

	outfile.close()




def loadWeights(layerNumber,numberOfTrainingData):
	file_path = path.relpath("weights/"+numberOfTrainingData+"."+layerNumber+".cvs")

	
	infile = open(file_path, "rb")
		
	return pickle.load(infile)
	

