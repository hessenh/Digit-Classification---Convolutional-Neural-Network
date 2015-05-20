from sklearn.datasets import fetch_mldata
import random

mnist = fetch_mldata('MNIST original')
# mnist.data
# mnist.taget



def getTrainingImageNumberList(numberOfImages):
	return random.sample(range(59999), numberOfImages)

def getImageAndTarget(index):
	data = mnist.data[index]
	d = []
	for i in range(0,len(data)):
		d.append(data[i])
		if(i%28==0):
			d.append(1)

	for i in range(0,29):
		d.append(1)

	t = []
	for i in range(0,10):
		if(mnist.target[index]==i):
			t.append(1)
		else:
			t.append(-1)
	return d,t

def getRandomImage():
	# Image
	r  = random.randrange(1,70000)
	data = mnist.data[r]
	d = []
	for i in range(0,len(data)):
		d.append(data[i])
		if(i%28==0):
			d.append(1)

	for i in range(0,29):
		d.append(1)

	# Target 
	t = []
	for i in range(0,10):
		if(mnist.target[r]==i):
			t.append(1)
		else:
			t.append(-1)
	return d,t

def getImages(size):
	images = []
	imagesTarget = []
	for i in range(size):
		img,imgTarget = getRandomImage()
		images.append(img)
		imagesTarget.append(imgTarget)
	return images,imagesTarget


