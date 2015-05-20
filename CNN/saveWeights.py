import pickle
from nnWeight import NNWeight



def saveWeights(layerNumber,numberOfTrainingData,weights):
	outfile = open(layerNumber+"."+numberOfTrainingData+".cvs", "wb")

	pickle.dump(weights, outfile)

	outfile.close()




def loadWeights(layerNumber,numberOfTrainingData):
	infile = open(layerNumber+"."+numberOfTrainingData+".cvs", "rb")

	return pickle.load(infile)



