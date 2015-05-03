import loadData,cnnConvolve,generateFilters,cnnPool as pool
from matplotlib import pyplot as plt, numpy as np



kernels = generateFilters.getKernels(1,5)
images = loadData.getImages(100)


#Convolve images with filters
cImages = cnnConvolve.convolveImages(images,kernels)


#Display the conved images
for i in range(0,0):
	imgplot = plt.imshow(cImages[i])
	plt.show()

#Average pooling
pooledFeauters = pool.averagePool(10,cImages)

#Dispaly pooling
for i in range(0,2	):
	imgplot = plt.imshow(pooledFeauters[i])
	plt.show()