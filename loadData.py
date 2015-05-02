from sklearn.datasets import fetch_mldata
import random

mnist = fetch_mldata('MNIST original')


def getRandomImage():
	r  = random.randrange(1,70000)
	data = mnist.data[r]

	newData = []
	index = 0;
	for i in range(0,28):
		temp = []
		for j in range(0,28):
			temp.append(data[index])
			index+=1
		newData.append(temp)

	return newData


def getImages(size):
	images = []
	for i in range(size):
		img = getRandomImage()
		images.append(img)

	return images