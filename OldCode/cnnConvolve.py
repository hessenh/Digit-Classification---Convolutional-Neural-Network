import scipy.signal as ss,numpy as np

def convolveImage(image, kernel):

    ci = ss.convolve2d(image, kernel, mode='valid')
    
    return ci    


def convolveImages(images,filters):

    result = []

    for img in images:
        for kernel in filters:
            cImg = convolveImage(img,kernel)
            cImg = sigmoid(cImg)

            result.append(cImg)

    return result

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
    #return 1.0 / (1.0 + np.exp(-1.0 * x))