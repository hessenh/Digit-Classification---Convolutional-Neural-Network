from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show, axes, sci
from matplotlib import cm, colors
from matplotlib.font_manager import FontProperties
from numpy import amin, amax, ravel
from numpy.random import rand


def chunck(l,n):
	return [l[i:i + n] for i in range(0, len(l), n)]


# Get in right format
def showImage(data,layer,filterNumber,filterSize):
	#print "Layer:",layer,"Filter:",filterNumber+1
	
	n = []
	for i in range(0,filterSize):
		t = []
		for j in range(0,filterSize):
			t.append(data[i*filterSize+j])
		n.append(t)

	return n
	#display(n)

#
# Shows the layers neurons
#
def showLayer(neurons,filterSize,row,col):
	l = chunck(neurons,filterSize*filterSize)

	for i in range(0,len(l)):
		for j in range(0,len(l[i])):
			l[i][j] = l[i][j].output

	filters = []
	for i in range(len(l)):
		filters.append(showImage(l[i],"1",i,filterSize))

	display(filters,row,col)

#
# Visualises the network
#
def visualise(nn):
	print "Visualising starting"

	showLayer(nn.layers[1].neurons,13,2,3)

	showLayer(nn.layers[2].neurons,5,5,10)


#
# To display the network to the client // BETA ;)
# 
def getNeuronOutputs(nn):
	neurons = nn.layers[1].neurons
	filterSize = 13
	row = 2
	col = 3

	l = chunck(neurons,filterSize*filterSize)

	for i in range(0,len(l)):
		for j in range(0,len(l[i])):
			l[i][j] = l[i][j].output

	filters = []
	for i in range(len(l)):
		filters.append(showImage(l[i],"1",i,filterSize))


	return filters

#
# Displays the data
#
def display(data,row,col):
	fig = plt.figure()
	
	for i in range(0,len(data)):
		lum_img = data[i]
		a=fig.add_subplot(col,row,i)
		imgplot = plt.imshow(lum_img)
		a.set_title(str(i+1))

	show()