from sklearn.datasets import fetch_mldata
import random

mnist = fetch_mldata('MNIST original')
# mnist.data
# mnist.taget

def getRandomImage():
	r  = random.randrange(1,70000)
	data = mnist.data[r]
	d = []
	for i in range(0,len(data)):
		d.append(data[i])
		if(i%28==0):
			d.append(1)

	for i in range(0,29):
		d.append(1)
	return d,mnist.target[r]

def getImages(size):
	images = []
	imagesTarget = []
	for i in range(size):
		img,imgTarget = getRandomImage()
		images.append(img)
		imagesTarget.append(imgTarget)
	return images,imagesTarget
