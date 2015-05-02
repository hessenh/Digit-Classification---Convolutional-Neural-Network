import cPickle, gzip, numpy as np,Image,cnnConvolve
from matplotlib import pyplot as plt

# Load the dataset
f = gzip.open('Dataset/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
data = train_set[0][23]

newData = []
index = 0;
for i in range(0,28):
	temp = []
	for j in range(0,28):
		temp.append(data[index])
		index+=1
	newData.append(temp)

data = np.array(newData)

imgplot = plt.imshow(data)
plt.show()


f.close()