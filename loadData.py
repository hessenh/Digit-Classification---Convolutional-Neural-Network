from sklearn.datasets import fetch_mldata
import random

mnist = fetch_mldata('MNIST original')
# mnist.data
# mnist.taget

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

	return newData,mnist.target[r]


def getImages(size):
	images = []
	imagesTarget = []
	for i in range(size):
		img,imgTarget = getRandomImage()
		images.append(img)
		imagesTarget.append(imgTarget)
	return images,imagesTarget