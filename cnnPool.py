def cnnPool(poolDim, convolvedFeatures):

	numImages = size(convolvedFeatures, 4);
	numFilters = size(convolvedFeatures, 3);
	convolvedDim = size(convolvedFeatures, 1);

	pooledFeatures = zeros(convolvedDim / poolDim, ...
	convolvedDim / poolDim, numFilters, numImages);

	

