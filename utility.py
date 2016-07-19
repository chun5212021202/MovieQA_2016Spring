
import numpy as np
from random import shuffle
import json
from pprint import pprint

import time


#########################################################
#														#
#						FUNCTIONS						#
#														#
#########################################################




def findLongestSentenceNum(plotFilePath, qaPathTrain, qaPathVal):

	with open( qaPathTrain ) as data_file:    
		train_q = json.load(data_file)

	with open( qaPathVal ) as data_file:    
		val_q = json.load(data_file)

	longest_train = 0
	longest_sentence_train = 0
	for question in train_q:
		qid = plotFilePath+question["qid"]+".wiki.json"
		with open(qid) as data_file:
			P1 = json.load(data_file)
			if len(P1) >= longest_train:
				longest_train = len(P1)
			for sentence in P1 :
				if len(sentence) >= longest_sentence_train:
					longest_sentence_train = len(sentence)

	print 'train longest paragraph : ', longest_train
	print 'train longest sentence : ', longest_sentence_train

	longest_val = 0
	longest_sentence_val = 0
	for question in val_q:
		qid = plotFilePath+question["qid"]+".wiki.json"
		with open(qid) as data_file:
			P2 = json.load(data_file)
			if len(P2) >= longest_val:
				longest_val = len(P2)
			for sentence in P2 :
				if len(sentence) >= longest_sentence_val:
					longest_sentence_val = len(sentence)

	print 'val longest paragraph : ', longest_val
	print 'val longest sentence : ', longest_sentence_val

	if longest_val > longest_train:
		longest = longest_val
	else :
		longest = longest_train

	if longest_sentence_val > longest_sentence_train:
		longest_sentence = longest_sentence_val
	else :
		longest_sentence = longest_sentence_train

	return longest, longest_sentence_train

def Train(nn_object, epoch, plotFilePath, qaPathTrain, qaPathVal, longest, longest_sentence) :
	#plotFilePath = 'plots_vec_sentence/'
	#qaPathTrain = "data/qa.mini_train.json"
	#qaPathVal = "data/qa.mini_val.json"

	with open( qaPathTrain ) as data_file:    
		train_q = json.load(data_file)

	with open( qaPathVal ) as data_file:    
		val_q = json.load(data_file)


	cost_count = []
	accuracy_test = []
	accuracy_train = []

	initial_test = Test(val_q, plotFilePath, nn_object, longest, longest_sentence)
	initial_train = Test(train_q, plotFilePath, nn_object, longest, longest_sentence)

	for turns in range(epoch) :

		cost_count.append(0)
		for question in train_q:
			qid = plotFilePath+question["qid"]+".wiki.json"
			with open(qid) as data_file:
				P = json.load(data_file)

			for sentence in P:
				sentence += [[0]*300]*(longest_sentence-len(sentence))

			P = P + [[[0]*300]*longest_sentence]*(longest-len(P))
			Q = question["question"]   # same as answer                   [word, word]
			A = question["answers"][0] # answer 0 is A  only one sentence [word, word]
			B = question["answers"][1]
			C = question["answers"][2]
			D = question["answers"][3]
			E = question["answers"][4]
			if not A :
				A = [[0]*300]
			if not B :
				B = [[0]*300]
			if not C :
				C = [[0]*300]
			if not D :
				D = [[0]*300]
			if not E :
				E = [[0]*300]	
			AnsOption = [0,0,0,0,0]
			AnsOption[question["correct_index"]] = 1 
		
			

			start = time.time()
			temp = nn_object.train( P, Q, A, B, C, D, E, AnsOption )
			cost_count[turns]+=temp[0]
			print turns,'-',question["qid"],': ',temp[0], " >>> time:",time.time()-start#, " >>> MaxGrad:",temp[1]
			print

			
		accuracy_test.append(Test(val_q, plotFilePath, nn_object, longest, longest_sentence))
		accuracy_train.append(Test(train_q, plotFilePath, nn_object, longest, longest_sentence))
		print 'total cost: ',cost_count
		print 'Total Correct Test: ', accuracy_test, ' / ', len(val_q)
		print 'Total Correct Train: ', accuracy_train, ' / ', len(train_q)
		print 'Initial Correct Test: ', initial_test, ' / ', len(val_q)
		print 'Initial Correct Train: ', initial_train, ' / ', len(train_q)





def Test(train_val_data, plotFilePath, nn_object, longest, longest_sentence) :	# train one epoch

	print ' *** START TESTING *** '

	correct = 0
	for question in train_val_data:
		qid = plotFilePath+question["qid"]+".wiki.json"
		with open(qid) as data_file:
			P = json.load(data_file)
		for sentence in P:
			sentence += [[0]*300]*(longest_sentence-len(sentence))
			

		P = P + [[[0]*300]*longest_sentence]*(longest-len(P))
		Q = question["question"]   # same as answer                   [word, word]
		A = question["answers"][0] # answer 0 is A  only one sentence [word, word]
		B = question["answers"][1]
		C = question["answers"][2]
		D = question["answers"][3]
		E = question["answers"][4]
		if not A :
			A = [[0]*300]
		if not B :
			B = [[0]*300]
		if not C :
			C = [[0]*300]
		if not D :
			D = [[0]*300]
		if not E :
			E = [[0]*300]	

		AnsOption = [0,0,0,0,0]
		AnsOption[question["correct_index"]] = 1

		
		#print np.asarray(P).shape, np.asarray(Q).shape, np.asarray(A).shape, np.asarray(B).shape, np.asarray(C).shape, np.asarray(D).shape, np.asarray(E).shape
	
		
			
		temp = nn_object.test( P, Q, A, B, C, D, E )
		print temp[0]

		if temp[1] == question["correct_index"] :
			correct+=1

	return correct




