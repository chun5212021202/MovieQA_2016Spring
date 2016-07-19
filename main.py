import theano
import numpy as np
import theano.tensor as T
import utility
import attentionNN
import time

start = time.time()


#Merge
EPOCH = 25
LEARNING_RATE = 0.001
ALPHA = 0.99

#LSTM
LSTM_DEPTH = 1	#unfunctional
LSTM_X_DIMENSION = 300
LSTM_Y_DIMENSION = 64
ATTENTION_KERNEL_WIDTH = 2

plotFilePath = '../data/5_sentencePlots_vec/'#'../data/plots_vec_sentence/'
qaPathTrain = "../data/qa.mini_train.json"
qaPathVal = "../data/qa.mini_val.json"



longest = utility.findLongestSentenceNum(plotFilePath, qaPathTrain, qaPathVal)

neural_network = attentionNN.NeuralNetwork(
	LSTM_DEPTH,  
	LSTM_X_DIMENSION, 
	LSTM_Y_DIMENSION,
	LEARNING_RATE, 
	ALPHA,
	ATTENTION_KERNEL_WIDTH,
	longest[0]
	)

print
print 'version: Atteneion'
print 'NeuralNetwork Construction Time >>>>>',time.time() - start, 'sec'
print


utility.Train(neural_network, EPOCH, plotFilePath, qaPathTrain, qaPathVal, longest[0], longest[1])
