import loadData,cnnConvolve,generateFilters
from matplotlib import pyplot as plt, numpy as np



kernels = generateFilters.getKernels(1,5)
images = loadData.getImages(1)


#Convolve images with filters
cImages = cnnConvolve.convolveImages(images,kernels)


#Display the conved images
for i in range(0,1):
	imgplot = plt.imshow(cImages[i])
	plt.show()