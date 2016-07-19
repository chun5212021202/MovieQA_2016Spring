import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

import numpy as np
from itertools import izip

class LSTM (object ):
	def __init__ (self, input_dimension, output_dimension):	# input_dimension should be 2*x_dimension

		self.ParameterInit(input_dimension, output_dimension)




	def PlusNode(self, Para1, Para2):
		return Para1 + Para2

	def MultiplyNode(self, Para1, Para2):
		return Para1 * Para2

	def Sigmoid(self, z):
		return 1 / (1 + T.exp(-z))

	def Tanh(self, z):
		return T.tanh(z)

	def ParameterInit(self, input_dimension, output_dimension):
		self.W = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension, output_dimension+input_dimension)),dtype = theano.config.floatX))
		self.B = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension)),dtype = theano.config.floatX))
		self.Wf = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension, output_dimension+input_dimension)),dtype = theano.config.floatX))
		self.Bf = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension)),dtype = theano.config.floatX))
		self.Wi = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension, output_dimension+input_dimension)),dtype = theano.config.floatX))
		self.Bi = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension)),dtype = theano.config.floatX))
		self.Wo = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension, output_dimension+input_dimension)),dtype = theano.config.floatX))
		self.Bo = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension)),dtype = theano.config.floatX))


	def InputGate(self, Control, Flow):		#flow: x, h_t-1
		_i = self.Sigmoid( T.dot(self.Wi, Control) + self.Bi.dimshuffle(0,'x') )
		_input = self.Tanh( T.dot(self.W, Flow) + self.B.dimshuffle(0,'x') )
		return self.MultiplyNode(_i, _input)

	def OutputGate(self, Control, Flow):	#flow: memory
		_o = self.Sigmoid( T.dot(self.Wo, Control) + self.Bo.dimshuffle(0,'x') )
		_input = self.Tanh(Flow)
		return self.MultiplyNode(_o, _input)

	def ForgetGate(self, Control, Flow):
		_f = self.Sigmoid( T.dot(self.Wf, Control) + self.Bf.dimshuffle(0,'x') )
		_input = Flow
		return self.MultiplyNode(_f, _input)


	def Get_C_and_H(self, x_t, h_tm1, c_tm1):
		signal = T.concatenate([h_tm1, x_t],axis=0)
		c_t = self.PlusNode( self.InputGate(signal, signal), self.ForgetGate(signal, c_tm1) )
		h_t = self.OutputGate( signal, c_t )
		return h_t, c_t
			





class NeuralNetwork (object):
	def __init__ (self, depth, x_dimension, feature_dimension, learning_rate, alpha, Qkernel_width, longest):
		self.LR = learning_rate
		self.ALPHA = alpha

		P = T.tensor3(dtype=theano.config.floatX)
		Q = T.matrix(dtype=theano.config.floatX)
		A = T.matrix(dtype=theano.config.floatX)
		B = T.matrix(dtype=theano.config.floatX)
		C = T.matrix(dtype=theano.config.floatX)
		D = T.matrix(dtype=theano.config.floatX)
		E = T.matrix(dtype=theano.config.floatX)

		P_seq = P.dimshuffle(1,2,0)
		Q_seq = Q.dimshuffle(0,1,'x')
		A_seq = A.dimshuffle(0,1,'x')
		B_seq = B.dimshuffle(0,1,'x')
		C_seq = C.dimshuffle(0,1,'x')
		D_seq = D.dimshuffle(0,1,'x')
		E_seq = E.dimshuffle(0,1,'x')
		P_seq_inverse = P_seq[::-1]
		Q_seq_inverse = Q_seq[::-1]
		A_seq_inverse = A_seq[::-1]
		B_seq_inverse = B_seq[::-1]
		C_seq_inverse = C_seq[::-1]
		D_seq_inverse = D_seq[::-1]
		E_seq_inverse = E_seq[::-1]

		y_hat_seq = T.vector("y")


		self.Network = LSTM(x_dimension, feature_dimension)
		self.Network_inverse = LSTM(x_dimension, feature_dimension)
		self.Network_A = LSTM(x_dimension, feature_dimension)
		self.Network_A_inverse = LSTM(x_dimension, feature_dimension)	
		'''
		self.Network_P = LSTM(x_dimension, feature_dimension)
		self.Network_Q = LSTM(x_dimension, feature_dimension)
		self.Network_A = LSTM(x_dimension, feature_dimension)
		self.Network_P_inverse = LSTM(x_dimension, feature_dimension)
		self.Network_Q_inverse = LSTM(x_dimension, feature_dimension)
		self.Network_A_inverse = LSTM(x_dimension, feature_dimension)		
		'''

		def step(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output

		def step_inverse(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network_inverse.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network_inverse.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output

		def A_step(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network_A.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network_A.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output

		def A_inverse_step(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network_A_inverse.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network_A_inverse.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output
		'''
		def P_step(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output

		def Q_step(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output

		def A_step(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output

		def P_inverse_step(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output

		def Q_inverse_step(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output

		def A_inverse_step(x_t, h_out_tm1, c_out_tm1):
			h_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]
			return h_output, c_output	
		'''


		c_init_P = theano.shared(np.array( np.zeros((feature_dimension,longest)), dtype = theano.config.floatX))
		h_init_P = T.zeros_like(c_init_P)
		c_init_Q = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_Q = T.zeros_like(c_init_Q)
		c_init_A = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_A = T.zeros_like(c_init_A)
		c_init_B = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_B = T.zeros_like(c_init_B)
		c_init_C = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_C = T.zeros_like(c_init_C)
		c_init_D = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_D = T.zeros_like(c_init_D)
		c_init_E = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_E = T.zeros_like(c_init_E)
		c_init_P_inverse = theano.shared(np.array( np.zeros((feature_dimension,longest)), dtype = theano.config.floatX))
		h_init_P_inverse = T.zeros_like(c_init_P_inverse)
		c_init_Q_inverse = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_Q_inverse = T.zeros_like(c_init_Q_inverse)
		c_init_A_inverse = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_A_inverse = T.zeros_like(c_init_A_inverse)
		c_init_B_inverse = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_B_inverse = T.zeros_like(c_init_B_inverse)
		c_init_C_inverse = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_C_inverse = T.zeros_like(c_init_C_inverse)
		c_init_D_inverse = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_D_inverse = T.zeros_like(c_init_D_inverse)
		c_init_E_inverse = theano.shared(np.array( np.zeros((feature_dimension,1)), dtype = theano.config.floatX))
		h_init_E_inverse = T.zeros_like(c_init_E_inverse)


		[h_seq_P, c_seq_P],_ = theano.scan(	# frame number, depth, h_dimension
			step,
			sequences = P_seq,
			outputs_info = [h_init_P, c_init_P]
			)
		[h_seq_Q, c_seq_Q],_ = theano.scan(	# frame number, depth, h_dimension
			step,
			sequences = Q_seq,
			outputs_info = [h_init_Q, c_init_Q]
			)
		[h_seq_A, c_seq_A],_ = theano.scan(	# frame number, depth, h_dimension
			A_step,
			sequences = A_seq,
			outputs_info = [h_init_A, c_init_A]
			)
		[h_seq_B, c_seq_B],_ = theano.scan(	# frame number, depth, h_dimension
			A_step,
			sequences = B_seq,
			outputs_info = [h_init_B, c_init_B]
			)
		[h_seq_C, c_seq_C],_ = theano.scan(	# frame number, depth, h_dimension
			A_step,
			sequences = C_seq,
			outputs_info = [h_init_C, c_init_C]
			)
		[h_seq_D, c_seq_D],_ = theano.scan(	# frame number, depth, h_dimension
			A_step,
			sequences = D_seq,
			outputs_info = [h_init_D, c_init_D]
			)
		[h_seq_E, c_seq_E],_ = theano.scan(	# frame number, depth, h_dimension
			A_step,
			sequences = E_seq,
			outputs_info = [h_init_E, c_init_E]
			)
		[h_seq_P_inverse, c_seq_P_inverse],_ = theano.scan(	# frame number, depth, h_dimension
			step_inverse,
			sequences = P_seq_inverse,
			outputs_info = [h_init_P_inverse, c_init_P_inverse]
			)
		[h_seq_Q_inverse, c_seq_Q_inverse],_ = theano.scan(	# frame number, depth, h_dimension
			step_inverse,
			sequences = Q_seq_inverse,
			outputs_info = [h_init_Q_inverse, c_init_Q_inverse]
			)
		[h_seq_A_inverse, c_seq_A_inverse],_ = theano.scan(	# frame number, depth, h_dimension
			A_step_inverse,
			sequences = A_seq_inverse,
			outputs_info = [h_init_A_inverse, c_init_A_inverse]
			)
		[h_seq_B_inverse, c_seq_B_inverse],_ = theano.scan(	# frame number, depth, h_dimension
			A_step_inverse,
			sequences = B_seq_inverse,
			outputs_info = [h_init_B_inverse, c_init_B_inverse]
			)
		[h_seq_C_inverse, c_seq_C_inverse],_ = theano.scan(	# frame number, depth, h_dimension
			A_step_inverse,
			sequences = C_seq_inverse,
			outputs_info = [h_init_C_inverse, c_init_C_inverse]
			)
		[h_seq_D_inverse, c_seq_D_inverse],_ = theano.scan(	# frame number, depth, h_dimension
			A_step_inverse,
			sequences = D_seq_inverse,
			outputs_info = [h_init_D_inverse, c_init_D_inverse]
			)
		[h_seq_E_inverse, c_seq_E_inverse],_ = theano.scan(	# frame number, depth, h_dimension
			A_step_inverse,
			sequences = E_seq_inverse,
			outputs_info = [h_init_E_inverse, c_init_E_inverse]
			)

		#h_seq_P_inverse = h_seq_P_inverse[::-1]
		#h_seq_Q_inverse = h_seq_Q_inverse[::-1]
		#h_seq_A_inverse = h_seq_A_inverse[::-1]
		#h_seq_B_inverse = h_seq_B_inverse[::-1]
		#h_seq_C_inverse = h_seq_C_inverse[::-1]
		#h_seq_D_inverse = h_seq_D_inverse[::-1]
		#h_seq_E_inverse = h_seq_E_inverse[::-1]


		h_seq_P_combine = T.concatenate([h_seq_P[-1], h_seq_P_inverse[-1]],axis=0)#.dimshuffle('x','x', 1, 0)
		h_seq_Q_combine = T.concatenate([h_seq_Q[-1], h_seq_Q_inverse[-1]],axis=0)
		h_seq_A_combine = T.concatenate([h_seq_A[-1], h_seq_A_inverse[-1]],axis=0)
		h_seq_B_combine = T.concatenate([h_seq_B[-1], h_seq_B_inverse[-1]],axis=0)
		h_seq_C_combine = T.concatenate([h_seq_C[-1], h_seq_C_inverse[-1]],axis=0)
		h_seq_D_combine = T.concatenate([h_seq_D[-1], h_seq_D_inverse[-1]],axis=0)
		h_seq_E_combine = T.concatenate([h_seq_E[-1], h_seq_E_inverse[-1]],axis=0)
		w,h = h_seq_P_combine.shape


		
		### Attention Model ###
		self.W_kernel = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(2*feature_dimension, 2*feature_dimension, Qkernel_width)),dtype = theano.config.floatX))
		self.B_kernel = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(2*feature_dimension, Qkernel_width)),dtype = theano.config.floatX))

		kernel = self.sigmoid( T.dot(T.reshape(h_seq_Q_combine,(2*feature_dimension,)), self.W_kernel) + self.B_kernel ).dimshuffle('x','x', 0, 1)# W:(2*feature_dimension, 2*feature_dimension, Qkernel_width)
		_,_,h1,w1 = kernel.shape

		conv_out = conv2d(
            input=h_seq_P_combine.dimshuffle('x','x', 0, 1),
            filters=kernel,
            border_mode='full'
        )[:, :, 2*feature_dimension-1:2*feature_dimension, (w1-1)/2 : (1-w1)/2]

		_,_,h2,w2 = conv_out.shape
		attention_map = self.softmax_map( T.reshape(conv_out,(w2,)) )
		attention_feature = h_seq_P_combine*attention_map.dimshuffle('x',0)	#128,sentence-number  361,sentence-number

		'''
		pool_out = downsample.max_pool_2d(
            input=attention_feature,
            ds=(1,longest),
            ignore_border=False
        )
		'''
		
		#self.Wo = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(2*feature_dimension, 2*feature_dimension)),dtype = theano.config.floatX)) #361 is the longest word number
		#self.B = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(2*feature_dimension)),dtype = theano.config.floatX))



		final = T.reshape(T.sum(attention_feature,axis=1),(2*feature_dimension,))#self.sigmoid(T.dot(T.reshape(pool_out,(2*feature_dimension,)), self.Wo) + self.B)
		Va = T.reshape(h_seq_A_combine,(2*feature_dimension,))#[-1]
		Vb = T.reshape(h_seq_B_combine,(2*feature_dimension,))#[-1]
		Vc = T.reshape(h_seq_C_combine,(2*feature_dimension,))#[-1]
		Vd = T.reshape(h_seq_D_combine,(2*feature_dimension,))#[-1]
		Ve = T.reshape(h_seq_E_combine,(2*feature_dimension,))#[-1]

		absVa = ((T.sum(Va**2))**0.5) * ((T.sum(final**2))**0.5)
	 	absVb = ((T.sum(Vb**2))**0.5) * ((T.sum(final**2))**0.5)
	 	absVc = ((T.sum(Vc**2))**0.5) * ((T.sum(final**2))**0.5)
	 	absVd = ((T.sum(Vd**2))**0.5) * ((T.sum(final**2))**0.5)
	 	absVe = ((T.sum(Ve**2))**0.5) * ((T.sum(final**2))**0.5)

	 	y_seq = self.softmax( T.stack([T.dot(Va,final)/absVa,T.dot(Vb,final)/absVb,T.dot(Vc,final)/absVc,T.dot(Vd,final)/absVd,T.dot(Ve,final)/absVe]) )#self.softmax( T.stack([Va*final/absVa,Vb*final/absVb,Vc*final/absVc,Vd*final/absVd,Ve*final/absVe]) )
	 	cost = T.sum( (y_hat_seq * -T.log(y_seq)) )
		


		parameters = []
		gradients = []

		parameters.append(self.Network.W)
		parameters.append(self.Network.B)
		parameters.append(self.Network.Wi)
		parameters.append(self.Network.Bi)
		parameters.append(self.Network.Wo)
		parameters.append(self.Network.Bo)
		parameters.append(self.Network.Wf)
		parameters.append(self.Network.Bf)
		parameters.append(self.Network_inverse.W)
		parameters.append(self.Network_inverse.B)
		parameters.append(self.Network_inverse.Wi)
		parameters.append(self.Network_inverse.Bi)
		parameters.append(self.Network_inverse.Wo)
		parameters.append(self.Network_inverse.Bo)
		parameters.append(self.Network_inverse.Wf)
		parameters.append(self.Network_inverse.Bf)
		#parameters.append(self.Network_Q.W)
		#parameters.append(self.Network_Q.B)
		#parameters.append(self.Network_Q.Wi)
		#parameters.append(self.Network_Q.Bi)
		#parameters.append(self.Network_Q.Wo)
		#parameters.append(self.Network_Q.Bo)
		#parameters.append(self.Network_Q.Wf)
		#parameters.append(self.Network_Q.Bf)
		#parameters.append(self.Network_Q_inverse.W)
		#parameters.append(self.Network_Q_inverse.B)
		#parameters.append(self.Network_Q_inverse.Wi)
		#parameters.append(self.Network_Q_inverse.Bi)
		#parameters.append(self.Network_Q_inverse.Wo)
		#parameters.append(self.Network_Q_inverse.Bo)
		#parameters.append(self.Network_Q_inverse.Wf)
		#parameters.append(self.Network_Q_inverse.Bf)
		parameters.append(self.Network_A.W)
		parameters.append(self.Network_A.B)
		parameters.append(self.Network_A.Wi)
		parameters.append(self.Network_A.Bi)
		parameters.append(self.Network_A.Wo)
		parameters.append(self.Network_A.Bo)
		parameters.append(self.Network_A.Wf)
		parameters.append(self.Network_A.Bf)
		parameters.append(self.Network_A_inverse.W)
		parameters.append(self.Network_A_inverse.B)
		parameters.append(self.Network_A_inverse.Wi)
		parameters.append(self.Network_A_inverse.Bi)
		parameters.append(self.Network_A_inverse.Wo)
		parameters.append(self.Network_A_inverse.Bo)
		parameters.append(self.Network_A_inverse.Wf)
		parameters.append(self.Network_A_inverse.Bf)
		parameters.append(self.W_kernel)
		parameters.append(self.B_kernel)


		#gradients = T.grad(cost,parameters)#T.clip(T.grad(cost,parameters),-0.2, 0.2)
		gradients.append(T.clip(T.grad(cost,parameters[0]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[1]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[2]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[3]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[4]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[5]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[6]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[7]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[8]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[9]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[10]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[11]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[12]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[13]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[14]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[15]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[16]),-0.2, 0.2))
		gradients.append(T.clip(T.grad(cost,parameters[17]),-0.2, 0.2))


		#max_gradients = T.max(T.stack([T.max(gradients[0]),T.max(gradients[1]),T.max(gradients[2]),T.max(gradients[3]),T.max(gradients[4]),T.max(gradients[5]),T.max(gradients[6]),T.max(gradients[7]),T.max(gradients[8]),T.max(gradients[9]),T.max(gradients[10]),T.max(gradients[11]),T.max(gradients[12]),T.max(gradients[13]),T.max(gradients[14]),T.max(gradients[15]),T.max(gradients[16]),T.max(gradients[17])]))


		self.update_parameter = theano.function(
			inputs = [P,Q,A,B,C,D,E,y_hat_seq],
			updates = self.UpdateParameter_RMSprop(parameters, gradients),
			outputs = [cost],
			allow_input_downcast = True
			)


		self.predict = theano.function(
			inputs = [P,Q,A,B,C,D,E],
			outputs = [y_seq,np.argmax(y_seq,axis=0)],
			allow_input_downcast = True
			)




	def softmax(self, h):
		total = T.sum(T.exp(h),axis=0)
		return T.exp(h)/total

	def softmax_map(self, h):
		total = T.sum(T.exp(h))
		return T.exp(h)/total

	def sigmoid(self, z):
		return 1 / (1 + T.exp(-z))

	def UpdateParameter_RMSprop(self, parameter, gradient) :
		update = []
		for p,g in izip(parameter, gradient) :
			acc = theano.shared(p.get_value() * 0.)		# called once
			acc_new = self.ALPHA * acc + (1 - self.ALPHA) * g ** 2		# called once
        	
			scale = T.sqrt(acc_new + 1e-6)
			g = g/scale

			update += [(acc, acc_new)]
			update += [(p, p - self.LR*g)]

		return update

	def train (self, training_p_seq, training_q_seq, training_a_seq, training_b_seq, training_c_seq, training_d_seq, training_e_seq, training_y_seq):	# training_x_seq (frame, x_dim)		 training_y_seq (frame, y_dim)
		return self.update_parameter(training_p_seq, training_q_seq, training_a_seq, training_b_seq, training_c_seq, training_d_seq, training_e_seq, training_y_seq)



	def test (self, training_p_seq, training_q_seq, training_a_seq, training_b_seq, training_c_seq, training_d_seq, training_e_seq):
		return self.predict(training_p_seq, training_q_seq, training_a_seq, training_b_seq, training_c_seq, training_d_seq, training_e_seq)