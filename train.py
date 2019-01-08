################## CODE for generating video ###############

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np 
import argparse
import sys
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score

import pickle

import math


Writer = animation.writers['ffmpeg']
writer = Writer(fps = 10, bitrate = 1800)

fig = plt.figure()
grid = plt.GridSpec(6, 6, hspace=0.2, wspace=0.2)       

ax1 =  fig.add_subplot(grid[1:, 0:]) 	# Neural Net
ax3 = fig.add_subplot(grid[0, -1])		# Error curve


class MultiLayer_NeuralNet():
  
	def __init__(self, config):
		'''
		To initialise hyperparamters of the network as class
		variables.
		Initialise weights, biases, activations and preactivations
		'''
		self.__dict__.update(config)

		#Calculating total number of weights required

		# if self.pretrain=="false":
		self.initialise_weights_bias()
		self.total_no_of_weights = (self.input_layer_size)*self.sizes[0] +sum((self.sizes[i])*self.sizes[i+1] for i in range(len(self.sizes)-1))+(self.sizes[-1])*self.output_layer_size

		#Initialise activations and preactivations as zero
		self.create_a_h()

		#Defining dictonaries to call functions depending on hyperparam value
		self.active={'sigmoid':self.sigmoid,'tanh':self.tanh}
		self.d_active={'sigmoid':self.d_sigmoid,'tanh':self.d_tanh}
		self.loss_fn={'sq':self.square,'ce':self.cross_entropy}
		self.d_loss={'sq':self.d_square,'ce':self.d_cross_entropy}
		self.f_out={'sq':self.lin,'ce':self.softmax}
		self.solver={'gd':self.gradient_descent, 'momentum':self.momentum_func, 'nag':self.nag, 'adam':self.adam}
		
		self.epoch = 0
		self.step = 0	
		self.Loss_arr = []

	def save_weights(self):
	      with open(self.save_dir+'weights.plk', 'w') as f:
	             pickle.dump([self.weights,self.bias], f)

	def load_weights(self):
	      with open(self.save_dir+'weights.pkl') as f:
	              self.weights,self.bias = pickle.load(f)
	      return 0

  
	def sigmoid(self,x):
		#Sigmoid activation
		#Implemented interms  of tanh for increased stability
		return .5 * (1 + np.tanh(.5 * x))

	def d_sigmoid(self,x):
		#Derivative of sigmoid activation (In terms of the sigmoid function)
		return x*(1.0-x)

	def tanh(self,x):
		#Tanh activation : direct numpy function
		return np.tanh(x)

	def d_tanh(self,x):
		#Derivative of tanh activation in terms of tanh
		return (1-x**2)

	def cross_entropy(self,y):
		#Cross entropy loss function
		return -np.sum(y*np.log(self.h[-1]),axis=None)/y.shape[0]

	def square(self,y):
		# Squared error loss function
		return np.sum((y-self.h[-1])**2)/y.shape[0]

	def d_square(self,y):
		#outer layer derivative for squared loss with linear f_out
		return -(y-self.h[-1])

	def d_cross_entropy(self,y):
		#outer layer derivative for cross entropy with softmax f_out
		return -(y-self.h[-1])
	

	def lin(self,x):
		#linear output
		return x

	def softmax(self,x):
		#softmax output funtion
		#To avoid instability, common terms , largest value taken common and cancelled
		exps = np.exp(x - np.max(x))
		return exps / np.sum(exps)
	
	def initialise_weights_bias(self):
		#Funnction to initialise weights and biases to random values

		#Initialising seed (hyperparam)
		np.random.seed(self.seed)

		
		#To accomodate different network architectures, weights 
		#and biases initialised as a list of numpy arrays.
		self.weights = []
		self.bias = []

		#Since dw and db have same dim as weights and biases
		#same dimesions used, with zero initialisation
		self.dw = []
		self.db=[]
		
		#Iterating over each set of layers to initialise weights. 
		#Previous layer size stored
		old_size = self.input_layer_size
		
		for new_size in self.sizes:
			#weights initialise from random distribution with zero mean, and max,min
			#varying over sqrt of fan in
			self.weights.append(np.random.uniform(low=-1.0/np.sqrt(old_size), high=1.0/np.sqrt(old_size), size=([old_size, new_size])))
			self.dw.append(np.zeros([old_size,new_size]))

			#Biases initialised at zero.
			self.bias.append((np.zeros([new_size,1])))
			self.db.append((np.zeros([new_size,1])))

			old_size = new_size
		
		#Weights for output layer initialised separately
		self.weights.append(np.random.uniform(low=-1.0/np.sqrt(self.input_layer_size), high=1.0/np.sqrt(self.input_layer_size), size=([old_size, self.output_layer_size])))
		self.bias.append(np.zeros([self.output_layer_size,1]))

		#Bias for output layer
		self.db.append((np.zeros([self.output_layer_size,1])))
		self.dw.append(np.zeros([old_size,self.output_layer_size]))

		#Momentum,etc have same dim as weights., so their values copied
		#This is not a shallow copy
		self.v_w = list(self.dw)
		self.prev_v_w = list(self.dw)
		self.m_w = list(self.dw)
		self.temp_w = list(self.dw)

		#Momentum etc for bias
		self.v_b = list(self.db)
		self.prev_v_b = list(self.db)
		self.m_b = list(self.db)
		self.temp_b = list(self.db)

        
	def create_a_h(self):
		#Iinitialising as empty lists
		#Dimensions of elements will be decided by assignment operation
		self.a = [[] for i in range(1+self.num_hidden)]
		self.h = [[] for i in range(1+self.num_hidden)]	

	def compute_dw_db(self, X, y, W):

		dl_da_l = self.d_loss[self.loss](y)
		self.dw[-1] = np.dot(dl_da_l.T, self.h[-2])
		self.db[-1] = dl_da_l
		dl_dh_i = np.dot(dl_da_l, W[-1].T)

		i = self.num_hidden-1
		while i>0 :
			dl_da_i = np.multiply(dl_dh_i, self.d_active[self.activation](self.h[i]))
			self.dw[i] = np.dot(dl_da_i.T, self.h[i-1])
			self.db[i] = dl_da_i
			dl_dh_i = np.dot(dl_da_i, W[i].T)
			i-=1

		dl_da_0 = np.multiply(dl_dh_i, self.d_active[self.activation](self.h[0]))
		self.dw[0] = np.dot(dl_da_0.T, X)
		self.db[0] = dl_da_0
        
		return

	def grad(self, X, y, W, b):
		self.a[0] = np.dot(X, W[0])
		self.a[0] += b[0].T
		self.h[0] = self.active[self.activation](self.a[0])

		for i in range(len(self.sizes)-1):
			self.a[i+1] = np.dot(self.h[i], W[i+1])
			self.a[i+1] += b[i+1].T
			self.h[i+1] = self.active[self.activation](self.a[i+1])
	      
		self.a[-1] = np.dot(self.h[-2], W[-1])
		self.a[-1] += b[-1].T
		self.h[-1] = self.f_out[self.loss](self.a[-1])

		return 
      
	def nag(self,X,y):  
		t=0
		step=0

		W = list(self.dw)
		b = list(self.db)

		L = []
		L_val = []
		max_epoch=3
		f_train=open(self.expt_dir+'log_train.txt','w')
		# f_val=open(self.expt_dir+'log_val.txt','w')
		f_train.write("Testing\n")
		# f_val.write("Testing\n")
		
		
		while t<max_epoch:
			for k in range(self.num_hidden+1):
				self.v_w[k] = self.momentum * self.prev_v_w[k] 
				self.v_b[k] = self.momentum * self.prev_v_b[k] 

				W[k] = self.weights[k] - self.v_w[k]
				b[k] = self.bias[k] - self.v_b[k]

			self.grad(X[step * self.batch_size : (1+step) * self.batch_size, :], \
				y[step * self.batch_size : (1+step) * self.batch_size], W, b)

			self.compute_dw_db(X[step * self.batch_size : (1+step) * self.batch_size, :], y[step * self.batch_size : (1+step) * self.batch_size], W)
			self.L=self.loss_fn[self.loss](y[step * self.batch_size : (1+step) * self.batch_size])
			
			self.momentum_func()

			L.append(self.L)

			step+=1
			
			# if step%100==0:

			# 	f_train.write('Epoch {}, Step {}, Loss: {}, Error: {}, lr={}\n'.format(t,step,self.L,accuracy_score(np.argmax(self.h[-1], axis=1),np.argmax(y[step * self.batch_size : (1+step) * self.batch_size, :], axis=1)),self.lr))
			# 	self.feed_forward(X_val)
				# self.L=self.loss_fn[self.loss](y_val)
				# f_val.write('Epoch {}, Step {}, Loss: {}, Error: {}, lr={}\n'.format(t,step,self.L,accuracy_score(np.argmax(self.h[-1], axis=1),np.argmax(y_val, axis=1)),self.lr))
			

			if (step+1)*self.batch_size>X.shape[0]:

				t+=1
				step = 0
				print("L : {}".format(self.L))
			
			# self.grad(X_val, y_val, W, b)
			# self.L=self.loss_fn[self.loss](y_val)
			# L_val.append(self.L)

			if (step+1)*self.batch_size>X.shape[0]:
				print("Epoch:{}".format(t))
				t+=1
				step = 0
				print("L : {}".format(L[-1]))
				acc = accuracy_score(np.argmax(self.h[-1], axis=1),np.argmax(y_val, axis=1))
				print("\nAccuracy : {}\n".format(acc))
				# if self.anneal == "true":
				# 	if L[-1]<L_epoch:
				# 		L_epoch=L_val
				# 		self.save_weights()

				# 	else:
						
				# 		t-=1
				# 		self.load_weights()
				# 		L_val=L_epoch
				# 		self.lr/=2
				# 		print("lr:{}".format(self.lr))
				# else:
				# 	if L[-1]<L_epoch:

				# 		self.save_weights()
                        


		f_train.close()
		# f_val.close()
		# print(np.argmax(self.h[-1], axis=1), np.argmax(y_val, axis=1))
		# print("\nAccuracy : {}\n".format(accuracy_score(np.argmax(y_val, axis=1), np.argmax(self.h[-1], axis=1))))\

		# plt.plot(np.arange(len(L)),L)
		# plt.plot(np.arange(len(L)),L_val)

		np.savetxt('plots/Loss_train_{}_{}_{}.csv'.format(self.num_hidden, self.sizes[0],self.loss), L, delimiter=',', newline='\n')
		# np.savetxt('plots/Loss_val_{}_{}_{}.csv'.format(self.num_hidden, self.sizes[0],self.loss), L_val, delimiter=',', newline='\n')

		return 0

	def adam(self,t):
		#Function optimiser using method of ADAptive Moments
		#Keeps track of both momentum and update history
		#More importance to less updated feature

		#Setting epsilon to avoid blow up in denominator
		eps=1e-8
      
		for k in range(self.num_hidden):
			#Accumulating momentum
			self.m_w[k]=self.beta1*self.m_w[k]+(1-self.beta1)*self.dw[k].T 
			self.m_b[k]=self.beta1*self.m_b[k]+(1-self.beta1)*np.sum(self.db[k], axis=0).reshape(len(self.bias[k]),1)

			#Accumulating magnitude of update
			self.v_w[k]=self.beta2*self.v_w[k]+(1-self.beta2)*(self.dw[k].T)**2
			self.v_b[k]=self.beta2*self.v_b[k]+(1-self.beta2)*np.sum(self.db[k], axis=0).reshape(len(self.bias[k]),1)**2
			
			#Updating weights and biases with bias correction 
			#(simulataneously to avoid instability)
			self.weights[k]=self.weights[k]-(self.lr/np.sqrt(self.v_w[k]/(1-(self.beta2)**(t+1))+eps))*self.m_w[k]/(1-(self.beta1)**(t+1))
			self.bias[k]=self.bias[k]-(self.lr/np.sqrt(self.v_b[k]/(1-(self.beta2)**(t+1))+eps))*self.m_b[k]/(1-(self.beta1)**(t+1))

	def momentum_func(self):
		#Keeps track of majority direction of previous updates
		#Builds momentum over updates.

		for k in range(self.num_hidden):
			#Accumulating momentum and gradients
			self.v_w[k] = self.momentum*self.prev_v_w[k] + self.lr*self.dw[k].T
			self.v_b[k] = self.momentum*self.prev_v_b[k] + self.lr*np.sum(self.db[k], axis=0).reshape(len(self.bias[k]),1)

			#updating weights and biases
			self.weights[k] = self.weights[k]-self.v_w[k]
			self.bias[k] = self.bias[k]-self.v_b[k]
		
		#Saving present values to history
		self.prev_v_w = self.v_w
		self.prev_v_b = self.v_b
          
		return 0
  
	def gradient_descent(self):
		for k in range(self.num_hidden):
			self.weights[k]=self.weights[k]-self.dw[k].T*self.lr/self.batch_size
			n = len(self.bias[k])
			self.bias[k]=self.bias[k].reshape(n,1)-np.sum(self.db[k], axis=0).reshape(n,1)*self.lr/self.batch_size
		return 0
	
	''' 
	Trains the given network with the training data passed.
	Calls, feed forward, then backprop, and finally updates gradients depending 
	on the solver chosen among the hyperparams
	'''
	def solve_network(self,X,y):
		
		if self.opt=='nag':
			self.nag(X,y)
			return
    
		t, step = 0, 0
		L = []
		L_epoch=100000
		max_epoch, temp, self.beta1, self.beta2 = 3 , 0, 0.9, 0.999

		f_train=open(self.expt_dir+'log_train.txt','w')
		# f_val=open(self.expt_dir+'log_val.txt','w')
		f_train.write("Testing\n")
		# f_val.write("Testing\n")
		while t<max_epoch:

			self.feed_forward(X[step * self.batch_size : (1+step) * self.batch_size, :])
			self.backprop(X[step * self.batch_size : (1+step) * self.batch_size, :], y[step * self.batch_size : (1+step) * self.batch_size, :], 0)
			L.append(self.L)
		
			if self.opt == 'adam':
				self.adam((step+1)*(t+1))
			else :
				self.solver[self.opt]()

		
			#L.append(self.L)
			step = step+1

			
			# if step%100==0:

			# 	f_train.write('Epoch {}, Step {}, Loss: {}, Error: {}, lr={}\n'.format(t,step,self.L,accuracy_score(np.argmax(self.h[-1], axis=1),np.argmax(y[step * self.batch_size : (1+step) * self.batch_size, :], axis=1)),self.lr))
			# 	self.feed_forward(X_val)
			# 	self.L=self.loss_fn[self.loss](y_val)
				# f_val.write('Epoch {}, Step {}, Loss: {}, Error: {}, lr={}\n'.format(t,step,self.L,accuracy_score(np.argmax(self.h[-1], axis=1),np.argmax(y_val, axis=1)),self.lr))
			

			if (step+1)*self.batch_size>X.shape[0]:

				t+=1
				step = 0
				# if self.lr>5e-8: #lr>5e-6:
				# 	self.lr = self.lr/8 #self.lr/8 => 55% accuracy
				# L.append(self.L)
				# print(1-(self.beta2)**(t+1))
				print("L : {}".format(self.L))
			# self.feed_forward(X_val)
			# self.L=self.loss_fn[self.loss](y_val)

			# L_val.append(self.L)
			if (step+1)*self.batch_size>X.shape[0]:
				print("Epoch:{}".format(t))
				t+=1
				step = 0
				print("L : {}".format(L[-1]))
				acc = accuracy_score(np.argmax(self.h[-1], axis=1),np.argmax(y_val, axis=1))
				print("\nAccuracy : {}\n".format(acc))
				# if self.anneal == "true":
				# 	if L[-1]<L_epoch:
				# 		L_epoch=L_val
				# 		self.save_weights()

				# 	else:
						
				# 		t-=1
				# 		self.load_weights()
				# 		L_val=L_epoch
				# 		self.lr/=2
				# 		print("lr:{}".format(self.lr))
				# else:
				# 	if L[-1]<L_epoch:

				# 		self.save_weights()
                        


		f_train.close()
		# f_val.close()
		# print(np.argmax(self.h[-1], axis=1), np.argmax(y_val, axis=1))
		# print("\nAccuracy : {}\n".format(accuracy_score(np.argmax(y_val, axis=1), np.argmax(self.h[-1], axis=1))))\

		# plt.plot(np.arange(len(L)),L)
		# plt.plot(np.arange(len(L)),L_val)

		np.savetxt('plots/Loss_train_{}_{}_{}.csv'.format(self.num_hidden, self.sizes[0],self.loss), L, delimiter=',', newline='\n')
		# np.savetxt('plots/Loss_val_{}_{}_{}.csv'.format(self.num_hidden, self.sizes[0],self.loss), L_val, delimiter=',', newline='\n')

		# plt.show()
  
  #To compute the derivatives wrt weights and biases using chain rule.
	def backprop(self,X,y,lbda):
    
    	#Computing loss fucntion value
		self.L=self.loss_fn[self.loss](y)
		
		#Gradients at output layer
		dl_da_l = self.d_loss[self.loss](y)

		self.dw[-1] = np.dot(dl_da_l.T, self.h[-2]) + lbda*self.weights[-1].T
		self.db[-1] = dl_da_l
		dl_dh_i = np.dot(dl_da_l, self.weights[-1].T)

    	#Iterating over the hidden layers
		i = self.num_hidden-1
		while i>0 :
			dl_da_i = np.multiply(dl_dh_i, self.d_active[self.activation](self.h[i]))
			self.dw[i] = np.dot(dl_da_i.T, self.h[i-1]) + lbda*self.weights[i].T
			self.db[i] = dl_da_i
			dl_dh_i = np.dot(dl_da_i, self.weights[i].T)
			i-=1

    	#
		dl_da_0 = np.multiply(dl_dh_i, self.d_active[self.activation](self.h[0]))
		self.dw[0] = np.dot(dl_da_0.T, X) + lbda*self.weights[0].T
		self.db[0] = dl_da_0
        
		return
	'''
	Forward pass through the network
	Starting from input , preactivationns, activations over the layers,
	output predictions, and the loss function are computed
	'''
	def feed_forward(self, X):
		
		#Forward pass through the input layers.
		#Preactivation
		self.a[0] = np.dot(X, self.weights[0])
		self.a[0] += np.repeat(self.bias[0], self.a[0].shape[0],axis=0).reshape(self.a[0].shape[1],self.a[0].shape[0]).T
		
		#Activation function called depending on hyperparameter setting
		self.h[0] = self.active[self.activation](self.a[0])

		for i in range(len(self.sizes)-1):
	  		#Preactivation at (i+1)th layer
			self.a[i+1] = np.dot(self.h[i], self.weights[i+1])
			self.a[i+1] += np.repeat(self.bias[i+1],self.a[i+1].shape[0],axis=0).reshape(self.a[i+1].shape[1],self.a[i+1].shape[0]).T
	 		#Activation
			self.h[i+1] = self.active[self.activation](self.a[i+1])
	  
		#Output layer preactivation
		self.a[-1] = np.dot(self.h[-2], self.weights[-1])
		self.a[-1] += np.repeat(self.bias[-1],self.a[-1].shape[0],axis=0).reshape(self.a[-1].shape[1],self.a[-1].shape[0]).T

		#Output function decided based on loss function given
		self.h[-1] = self.f_out[self.loss](self.a[-1])

		return

	def draw_neural_net(self, ax, left, right, bottom, top, layer_sizes, grad, weights, values):
	    '''
	    Draw a neural network cartoon using matplotilb.
	    
	    :usage:
	        >>> fig = plt.figure(figsize=(12, 12))
	        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
	    
	    :parameters:
	        - ax : matplotlib.axes.AxesSubplot
	            The axes on which to plot the cartoon (get e.g. by plt.gca())
	        - left : float
	            The center of the leftmost node(s) will be placed here
	        - right : float
	            The center of the rightmost node(s) will be placed here
	        - bottom : float
	            The center of the bottommost node(s) will be placed here
	        - top : float
	            The center of the topmost node(s) will be placed here
	        - layer_sizes : list of int
	            List of layer sizes, including input and output dimensionality
	    '''
	    ax1.clear()
	    ax1.axis('off')

	    n_layers = len(layer_sizes)
	    v_spacing = (top - bottom)/float(max(layer_sizes))
	    h_spacing = (right - left)/float(len(layer_sizes) - 1)
	    # Nodes
	    for n, layer_size in enumerate(layer_sizes):
	        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
	        for m in xrange(layer_size):
	            # circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
	            #                     color='w', ec='k', zorder=4)

	            # # ax.add_artist(circle)
	            # label = ax.annotate("cpicpi", xy=(n*h_spacing + left, layer_top - m*v_spacing), fontsize=30,
	            #                     verticalalignment='center', horizontalalignment='center')
	            # # text = ax.annotate("v", xy=(n*h_spacing + left , layer_top - m*v_spacing), 
	            # #                     fontsize = 20, color = 'b')
	            # ax.add_patch(circle)

	            bbox_props = dict(boxstyle="circle,pad=0.3", fc="cyan", ec="b", lw=2)
	            t = ax1.text(n*h_spacing + left, layer_top - m*v_spacing, 
	                        "{}".format(round(values[n][m], 2)), ha="center", va="center", rotation=0,
	                        size=12,
	                        bbox=bbox_props)

	    # Edges
	    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
	        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
	        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
	        for m in xrange(layer_size_a):
	            for o in xrange(layer_size_b):
	                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
	                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
	                slope = ((layer_top_b - o*v_spacing) - (layer_top_a - m*v_spacing))/(h_spacing)
	                c_val = layer_top_a - m*v_spacing - slope*(n*h_spacing + left)
	                
	                theta = math.atan(slope)
	                x_val = (v_spacing/2)*math.cos(theta)

	                text_1 = ax1.text((n*h_spacing + left + x_val), 
	                                  slope*(n*h_spacing + left + x_val) + c_val, 
	                                  "{}".format(round(self.weights[n][m][o], 2)), fontsize = 10, color = 'g')

	                text_2 = ax1.text(((n + 1)*h_spacing + left - 3/2*x_val), 
	                                  slope*((n + 1)*h_spacing + left - 3/2*x_val) + c_val, 
	                                  "{}".format(round(grad[n][o][m], 2)), fontsize = 10, color = 'r')
	                ax1.add_artist(line)

	
	def plot_error(self):
	    ax3.clear()
	    ax3.plot(range(len(self.Loss_arr)), self.Loss_arr)

    # Function for animation
	def Animate(self,i):

		if self.opt=='nag':
			self.nag(X,y)
			return

		L_epoch=100000
		max_epoch, temp, self.beta1, self.beta2 = 3 , 0, 0.9, 0.999


		layer_val = list(self.sizes)
		layer_val.insert(0,self.input_layer_size)
		layer_val.append(self.output_layer_size)

		X = self.x_train
		y = self.y_train

		self.feed_forward(X[self.step * self.batch_size : (1+self.step) * self.batch_size, :])
		self.backprop(X[self.step * self.batch_size : (1+self.step) * self.batch_size, :], y[self.step * self.batch_size : (1+self.step) * self.batch_size, :], 0)
		self.Loss_arr.append(self.L)

		if self.opt == 'adam':
			self.adam((self.step+1)*(self.epoch+1))
		else :
			self.solver[self.opt]()

	
		self.step = self.step+1

		if (self.step+1)*self.batch_size>X.shape[0]:

			self.epoch+=1
			self.step = 0

		values = []
		values.append(X[self.step * self.batch_size, :])
		for p in range(len(self.h)):
			values.append(self.h[p][0])
		# print(values)

		self.draw_neural_net(fig.gca(), 0, 1, 0, 1, layer_val, self.dw, self.weights, values)


		self.plot_error()

# Function for reading the data and starting the training
def get_data(arg_val):
  	
	config = vars(arg_val)

  	# if config['pretrain'] == 'false':
	x_train = np.genfromtxt(config['train'], delimiter=',', missing_values="NaN", skip_header=-1).astype("float")
	# x_val = np.genfromtxt("scale_val.csv", delimiter=',', missing_values=".NaN", skip_header=-1).astype("float")

	y_train  = np.eye(2)[np.array(x_train[:, -1]).astype("int")]
	# y_val = np.eye(10)[np.array(x_val[:, -1]).astype("int")]

	x_train  = x_train[:, :-1]
	# x_val = x_val[:, 1:-1]
	config.update({"input_layer_size" : x_train.shape[1], "output_layer_size" : config['out']})

	config.update({"x_train" : x_train, "y_train" : y_train})
		
	# x_test = np.genfromtxt("scale_test.csv", delimiter=',', missing_values=".NaN", skip_header=-1).astype("float")
	# x_test = x_test[:, 1:]

	# Conversion to dictionary
	config["sizes"] = [int(s) for s in config["sizes"].split(',')]
	config["seed"] = 1234
	
  	# Configuring the neural network with the hyperparameters
	nn = MultiLayer_NeuralNet(config)
  
  	# Start the animation and call training inside
  	ani = animation.FuncAnimation(fig, nn.Animate, frames=100, interval=200)
	ani.save('Neu_vis.mp4', writer = writer)
	plt.show()
  	# Train, validate and test
	# if config["pretrain"]=='false':

	# nn.solve_network(x_train, y_train)


  

# Function to parse hyperparameters from the commandline
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action="store", dest="lr", default=1, type = float)
    parser.add_argument('--momentum', action="store", dest="momentum", default=0.5, type = float)
    parser.add_argument('--num_hidden', action="store", dest="num_hidden", default=1, type = int)
    parser.add_argument('--out', action="store", dest="out", default=2, type = int)
    parser.add_argument('--sizes', action="store", dest="sizes", default="2", type = str)
    parser.add_argument('--activation', action="store", dest="activation", default="sigmoid", type = str)
    parser.add_argument('--loss', action="store", dest="loss", default="sq", type = str)
    parser.add_argument('--opt', action="store", dest="opt", default="adam", type = str)
    parser.add_argument('--batch_size', action="store", dest="batch_size", default=5, type = int)
    # parser.add_argument('--anneal', action="store", dest="anneal")
    parser.add_argument('--save_dir', action="store", dest="save_dir", default="", type = str)
    parser.add_argument('--expt_dir', action="store", dest="expt_dir", default="", type = str)
    parser.add_argument('--train', action="store", dest="train", default="train.csv", type = str)
    parser.add_argument('--frames', action="store", dest="frames", default=100, type = int)
    # parser.add_argument('--val', action="store", dest="val", default="scale_val.csv", type = str)
    # parser.add_argument('--pretrain', action="store", dest="pretrain", default="false", type = str)

    arg_val = parser.parse_args()
  
  # Read the data and train the network
    get_data(arg_val)

parse()