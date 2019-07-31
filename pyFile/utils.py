import numpy as np
###############tsne##############
# import numpy as np
#i#mport pylab as Plot

def Hbeta(D = np.array([]), beta = 1.0):
	#"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta);
	sumP = sum(P);
	H = np.log(sumP) + beta * np.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
	#"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print ("Computing pairwise distances...")
	(n, d) = X.shape;
	sum_X = np.sum(np.square(X), 1);
	D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
	P = np.zeros((n, n));
	beta = np.ones((n, 1));
	logU = np.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print ("Computing P-values for point ", i, " of ", n, "...")

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf;
		betamax =  np.inf;
		Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	print ("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)));
	return P;


def pca(X = np.array([]), no_dims = 2):
	#"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print ("Preprocessing the data using PCA...")
	(n, d) = X.shape;
	X = X - np.tile(np.mean(X, 0), (n, 1));
	(l, M) = np.linalg.eig(np.dot(X.T, X));
	Y = np.dot(X, M[:,0:no_dims]);
	return Y;


def tsne(X = np.array([]), no_dims = 2, initial_dims = 10, perplexity = 30.0):
#     """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
# 	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	# Check inputs
# 	if isinstance(no_dims, float):
# 		print ("Error: array X should have type float.");
# 		return -1;
# 	if (round(no_dims) != no_dims):
# 		print ("Error: number of dimensions should be an integer.");
# 		return -1;

	# Initialize variables
	X = pca(X, initial_dims).real;
	(n, d) = X.shape;
	max_iter = 10000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = np.random.randn(n, no_dims);
	dY = np.zeros((n, no_dims));
	iY = np.zeros((n, no_dims));
	gains = np.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	P = P + np.transpose(P);
	P = P / np.sum(P);
	P = P * 4;									# early exaggeration
	P = np.maximum(P, 1e-12);

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = np.sum(np.square(Y), 1);
		num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / np.sum(num);
		Q = np.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if (iter < 20):
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - np.tile(np.mean(Y, 0), (n, 1));

		# Compute current value of cost function
		if ((iter + 1) % 10 == 0):
			C = np.sum(P * np.log(P / Q));
			print ("Iteration ", (iter + 1), ": error is ", C)

		# Stop lying about P-values
		if (iter == 100):
			P = P / 4;

	# Return solution
	return Y;
##########Model build Finished!############
def LoadTrainData(filename):
    trEmisDt = np.loadtxt(filename)
    return trEmisDt.tolist()


def save_model(model, tempPath, is_pytorch):
    if not is_pytorch:
        model.save(tempPath)

###########################################
def splitArr(arr, val):
    res = np.where(arr[:,1]==val)[0]
    return res
###########################################################################

def readVectorFromFile(vecfile, line_length):
    vectorFile = open(vecfile, 'r')
    wordVec = {}
    for line in vectorFile:
        # print(line)
        a = line.split()
        vec = []
        if (line_length != len(a)):
            print('readVectorFromFile Error!')
        for i in range(len(a) - 1):
            vec += [float(a[i + 1])]

        wordVec[a[0]] = vec
    vectorFile.close()
    return wordVec


#########################################################################

def loadData(filename):
    trEmisDt = np.loadtxt(filename)
    return trEmisDt.tolist()


def saveData(mydata, filename):
    np.savetxt(filename,mydata)
    # wf = open('a_file_2.conll', 'w')
    # for i in range(len(new_sentences)):
        # wf.write(str(new_sentences[i]))
    # wf.close()

def isnotnum(str):
    nums = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    for i in str:
        if i in nums:
            return False
    return True

def sentence_loader(file_h):
    w2i = {}
    sentences = [[j if isnotnum(j) else '<num>' for j in i.rstrip("\t").split('\t')] for i in file_h]
    id = 0
    for s in sentences:
        for i in s:
            if i not in w2i:
                w2i[i] = id
                id = id + 1
    w2i['<unknow>'] = id
    stcs = [[w2i[i] for i in s] for s in sentences]
    stcs_len = [len(j) for j in sentences]
    i2w = {k: i for i, k in w2i.items()}
    return w2i, i2w, stcs, stcs_len

def sentences2id(file_h, w2i, length):
    sentences = [[j if isnotnum(j) else '<num>' for j in i.rstrip("\t").split('\t')] for i in file_h if len(i.rstrip("\t").split('\t')) < length]
    stcs = [[w2i[i] if i in w2i else w2i['<unknow>'] for i in s] for s in sentences]
    stcs_len = [len(j) for j in sentences]
    return stcs, stcs_len