################## CODE for generating video ###############

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time
import numpy as np 
import argparse
import sys
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score

import pickle
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

import math

from activations import *


Writer = animation.writers['ffmpeg']
writer = Writer(fps = 10, bitrate = 1800)
img_count = 1

fig = plt.figure()
fig.patch.set_facecolor('black')
grid = plt.GridSpec(10, 10, hspace=0.2, wspace=0.2)

ax1 =  fig.add_subplot(grid[3:, 0:]) 	# Neural Net
ax3 = fig.add_subplot(grid[-1, -1])		# Error curve
ax3.tick_params(axis='x', colors='white')
ax3.tick_params(axis='y', colors='white')

ax_f1 = fig.add_subplot(grid[0:3, 0:-6])		# Computation
ax_f2 = fig.add_subplot(grid[0:3, -5])
ax_f3 = fig.add_subplot(grid[0:3, -3])
ax_f4 = fig.add_subplot(grid[0:3, -1])

ax_x = fig.add_subplot(grid[1:2, -6])
ax_equals = fig.add_subplot(grid[1:2, -4])
ax_sig = fig.add_subplot(grid[0:3, -2])

ax_loss = fig.add_subplot(grid[0:3, -7:-2])

ax_a1 = fig.add_subplot(grid[0:3, 0])	# y_hat
ax_a2 = fig.add_subplot(grid[1:2, 1])	# minus / dot
ax_a3 = fig.add_subplot(grid[0:3, 2])	# e(y)
ax_a4 = fig.add_subplot(grid[1:2, 3])	# =
ax_a5 = fig.add_subplot(grid[0:3, 4])	# da

ax_h5 = fig.add_subplot(grid[0:3, -1])		# dh
ax_h4 = fig.add_subplot(grid[1:2, -2])		# =
ax_h3 = fig.add_subplot(grid[0:3, -3])		# da
ax_h2 = fig.add_subplot(grid[1:2, -4])		# multiply
ax_h1 = fig.add_subplot(grid[0:3, 0:-5])	# dw

ax_w1 = fig.add_subplot(grid[0:3, 0])	# da
ax_w2 = fig.add_subplot(grid[1:2, 1])	# multiply
ax_w3 = fig.add_subplot(grid[1:2, 2:5])	# h
ax_w4 = fig.add_subplot(grid[1:2, 5])	# =
ax_w5 = fig.add_subplot(grid[0:3, 6:])	# dw

class MultiLayer_NeuralNet(active):
  
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
		self.solver={'gd':self.gradient_descent, 'momentum':self.momentum_func, 'adam':self.adam}
		
		self.epoch = 0
		self.step = 0	
		self.Loss_arr = []	
	
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
		self.da = [[] for i in range(1+self.num_hidden)]
		self.dh = [[] for i in range(self.num_hidden)]

	def compute_dw_db(self, X, y, W):

		self.da[-1] = self.d_loss[self.loss](y)
		self.dw[-1] = np.dot(self.da[-1].T, self.h[-2])
		self.db[-1] = self.da[-1]
		self.dh[-1] = np.dot(self.da[-1], W[-1].T)

		i = self.num_hidden-1
		while i>0 :
			self.da[i] = np.multiply(self.dh[i], self.d_active[self.activation](self.h[i]))
			self.dw[i] = np.dot(self.da[i].T, self.h[i-1])
			self.db[i] = self.da[i]
			self.dh[i] = np.dot(self.da[i], W[i].T)
			i-=1

		self.da[0] = np.multiply(self.dh[i], self.d_active[self.activation](self.h[0]))
		self.dw[0] = np.dot(self.da[0].T, X)
		self.db[0] = self.da[0]
        
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
  
	def gradient_descent(self):
		for k in range(self.num_hidden):
			self.weights[k]=self.weights[k]-self.dw[k].T*self.lr/self.batch_size
			n = len(self.bias[k])
			self.bias[k]=self.bias[k].reshape(n,1)-np.sum(self.db[k], axis=0).reshape(n,1)*self.lr/self.batch_size
		return 0

  #To compute the derivatives wrt weights and biases using chain rule.
	def backprop(self, X, y, lbda):
    
		#Gradients at output layer
		self.da[-1] = self.d_loss[self.loss](y)

		self.dw[-1] = np.dot(self.da[-1].T, self.h[-2]) + lbda*self.weights[-1].T
		self.db[-1] = self.da[-1]
		self.dh[-1] = np.dot(self.da[-1], self.weights[-1].T)

    	#Iterating over the hidden layers
		i = self.num_hidden-1
		while i>0 :
			self.da[i] = np.multiply(self.dh[i], self.d_active[self.activation](self.h[i]))
			self.dw[i] = np.dot(self.da[i].T, self.h[i-1]) + lbda*self.weights[i].T
			self.db[i] = self.da[i]
			self.dh[i] = np.dot(self.da[i], self.weights[i].T)
			i-=1

    	#
		self.da[0] = np.multiply(self.dh[0], self.d_active[self.activation](self.h[0]))
		self.dw[0] = np.dot(self.da[0].T, X) + lbda*self.weights[0].T
		self.db[0] = self.da[0]
        
		return
	'''
	Forward pass through the network
	Starting from input , preactivationns, activations over the layers,
	output predictions, and the loss function are computed
	'''
	def feed_forward(self, X, y):
		
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

		#Computing loss fucntion value
		self.L=self.loss_fn[self.loss](y)		
		return

	def clear_all(self, layer_num, fun1):
		ax1.clear()

		ax_f1.clear()
		ax_f2.clear()
		ax_f3.clear()
		ax_f4.clear()
		ax_sig.clear()
		ax_equals.clear()
		ax_x.clear()
		ax_loss.clear()

		ax_a1.clear()
		ax_a2.clear()
		ax_a3.clear()
		ax_a4.clear()
		ax_a5.clear()

		ax_h5.clear()
		ax_h4.clear()
		ax_h3.clear()
		ax_h2.clear()
		ax_h1.clear()

		ax_w1.clear()
		ax_w2.clear()
		ax_w3.clear()
		ax_w4.clear()
		ax_w5.clear()

		ax_f1.set_xticks([])
		ax_f2.set_xticks([])
		ax_f3.set_xticks([])
		ax_f4.set_xticks([])
		ax_x.set_xticks([])
		ax_equals.set_xticks([])
		ax_sig.set_xticks([])
		ax_loss.set_xticks([])
		ax_f1.set_yticks([])
		ax_f2.set_yticks([])
		ax_f3.set_yticks([])
		ax_f4.set_yticks([])
		ax_x.set_yticks([])
		ax_equals.set_yticks([])
		ax_sig.set_yticks([])
		ax_loss.set_yticks([])

		ax_a1.set_xticks([])
		ax_a2.set_xticks([])
		ax_a3.set_xticks([])
		ax_a4.set_xticks([])
		ax_a5.set_xticks([])

		ax_h5.set_xticks([])
		ax_h4.set_xticks([])
		ax_h3.set_xticks([])
		ax_h2.set_xticks([])
		ax_h1.set_xticks([])

		ax_w1.set_xticks([])
		ax_w2.set_xticks([])
		ax_w3.set_xticks([])
		ax_w4.set_xticks([])
		ax_w5.set_xticks([])

		ax_a1.set_yticks([])
		ax_a2.set_yticks([])
		ax_a3.set_yticks([])
		ax_a4.set_yticks([])
		ax_a5.set_yticks([])

		ax_h5.set_yticks([])
		ax_h4.set_yticks([])
		ax_h3.set_yticks([])
		ax_h2.set_yticks([])
		ax_h1.set_yticks([])

		ax_w1.set_yticks([])
		ax_w2.set_yticks([])
		ax_w3.set_yticks([])
		ax_w4.set_yticks([])
		ax_w5.set_yticks([])

		ax1.axis('off')
		ax_f1.axis('off')
		ax_f2.axis('off')
		ax_f3.axis('off')
		ax_f4.axis('off')
		ax_sig.axis('off')
		ax_equals.axis('off')
		ax_x.axis('off')
		ax_loss.axis('off')

		ax_a1.axis('off')
		ax_a2.axis('off')
		ax_a3.axis('off')
		ax_a4.axis('off')
		ax_a5.axis('off')

		ax_h5.axis('off')
		ax_h4.axis('off')
		ax_h3.axis('off')
		ax_h2.axis('off')
		ax_h1.axis('off')

		ax_w1.axis('off')
		ax_w2.axis('off')
		ax_w3.axis('off')
		ax_w4.axis('off')
		ax_w5.axis('off')

		if layer_num > 0:
			ax_f1.axis('on')
			ax_f2.axis('on')
			ax_f3.axis('on')
			ax_f4.axis('on')
			ax_x.axis('on')
			ax_equals.axis('on')
			ax_sig.axis('on')
			
		elif layer_num == None:
			ax_loss.axis('on')

		elif fun1 == 'a':
			ax_a1.axis('on')
			ax_a2.axis('on')
			ax_a3.axis('on')
			ax_a4.axis('on')
			ax_a5.axis('on')
		elif fun1 == 'w':
			ax_w1.axis('on')
			ax_w2.axis('on')
			ax_w3.axis('on')
			ax_w4.axis('on')
			ax_w5.axis('on')
		elif fun1 == 'h':
			ax_h5.axis('on')
			ax_h4.axis('on')
			ax_h3.axis('on')
			ax_h2.axis('on')
			ax_h1.axis('on')



	def draw_nodes(self, ax, left, right, bottom, top, layer_sizes, preactivations, 
					values, layer_num, neuron_num, v_spacing, h_spacing, n_layers, curr_y=None):
	    # Nodes
	    for n, layer_size in enumerate(layer_sizes):
	        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
	        for m in xrange(layer_size):
	            fc = "white"
	            if n == layer_num and m == neuron_num:
	            	fc = "green"
	            if layer_num == None and n == len(layer_sizes)-1:
	            	fc = "cyan"
	            bbox_props = dict(boxstyle="circle,pad=0.3", fc=fc, ec="w", lw=2)
	            t = ax1.text(n*h_spacing + left, layer_top - m*v_spacing, 
	                        "{}".format(round(values[n][m], 2)), ha="center", va="center", rotation=0,
	                        size=12,
	                        bbox=bbox_props)

	            if layer_num == None and n == len(layer_sizes)-1: # loss computation
	    			bbox_props = dict(boxstyle="square,pad=0.3", fc=fc, ec="w", lw=2)
	    			t1 = ax1.text(n*h_spacing + left + 0.12, layer_top - m*v_spacing, "{}".format(round(curr_y[0][m], 2)), ha="center", va="center", rotation=0, size=12, bbox=bbox_props)
	    			if curr_y[0][m] == 1:
	    				ax_loss.text(0.5, 0.5, r'$L = -log(\hat{y_{l}}) = -log($' + "{}".format(round(values[n][m], 2)) + ') = ' + "{}".format(round(self.L, 2)), va='center', ha='center', color = 'red')


	            if layer_num == n+1:
	            	ax_f2.text(0+0.5, layer_size-(m+0.5), str(round(values[n][m], 2)), va='center', ha='center', color='g')
	            if layer_num == n:
	            	m_color = 'black'
	            	if m == neuron_num:
	            		m_color = 'g'
	            	ax_f3.text(0+0.5, layer_size-(m+0.5), str(round(preactivations[n][m], 2)), va='center', ha='center', color=m_color)
	            	ax_f4.text(0+0.5, layer_size-(m+0.5), str(round(values[n][m], 2)), va='center', ha='center', color=m_color)

	        if n+1 == layer_num:
	        	ax_f2.set_xlim(0, 1)
	        	ax_f2.set_ylim(0, layer_size)

	        if n == layer_num:
	        	ax_f3.set_xlim(0, 1)
	        	ax_f3.set_ylim(0, layer_size)
	        	ax_f4.set_xlim(0, 1)
	        	ax_f4.set_ylim(0, layer_size)

	def draw_edges(self, ax, left, right, bottom, top, layer_sizes, preactivations, values, 
					layer_num, neuron_num, v_spacing, h_spacing, n_layers, curr_y=None):
		# Edges
		for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
			layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
			for m in xrange(layer_size_a):
				for o in xrange(layer_size_b):
					line_c = 'w'
					color = 'w'
					m_color = 'black'
					if layer_num == n+1:
						if o == neuron_num:
							color = 'g'
							line_c = 'g'
							m_color = 'g'

						w_val = round(self.weights[n][m][o], 2)
						ax_f1.text(m+0.5, layer_size_b -(o+0.5), str(w_val), va='center', ha='center', color=m_color)

						line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
						              [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c=line_c)
						slope = ((layer_top_b - o*v_spacing) - (layer_top_a - m*v_spacing))/(h_spacing)
						c_val = layer_top_a - m*v_spacing - slope*(n*h_spacing + left)

						theta = math.atan(slope)
						x_val = (v_spacing/2)*math.cos(theta)

						# if counter%2 == 0:	                
						text_1 = ax1.text((n*h_spacing + left + x_val), slope*(n*h_spacing + left + x_val) + c_val, "{}".format(round(self.weights[n][m][o], 2)), fontsize = 10, color = color)
						# else:
						# 	text_2 = ax1.text(((n + 1)*h_spacing + left - 3/2*x_val), slope*((n + 1)*h_spacing + left - 3/2*x_val) + c_val, "{}".format(round(grad[n][o][m], 2)), fontsize = 10, color = 'r')

						ax1.add_artist(line)

					elif (layer_num == -1*n):
						line_c = 'b'
						color = 'b'
						m_color = 'b'
						if fun1 == 'a':
							# show del a_k

						elif fun1 == 'h':
							# show del h_k

						elif fun1 == 'w':
							# show del w_k


					elif layer_num == -(len(layer_sizes)-1):
						# show del a_L 




				if layer_num == n+1:
					ax_f1.set_xlim(0, layer_size_a)
					ax_f1.set_ylim(0, layer_size_b)


	def draw_neural_net(self, ax, left, right, bottom, top, layer_sizes, 
						preactivations, values, layer_num, neuron_num, fun1=None, curr_y=None):
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
	    # print("draw i = ", counter)

	    self.clear_all(layer_num, fun1)

	    ax_x.text(0.5, 0.5, "*", va='center', ha='center')
	    ax_equals.text(0.5, 0.5, "=", va='center', ha='center')
	    if layer_num == len(layer_sizes)-1:
	    	ax_sig.text(0.5, 0.5, 'softmax', va='center', ha='center', fontsize='smaller', rotation=90)
	    elif layer_num > 0:
	    	ax_sig.text(0.5, 0.5, r'$\sigma$', va='center', ha='center')
	    n_layers = len(layer_sizes)
	    v_spacing = (top - bottom)/float(max(layer_sizes))
	    h_spacing = (right - left)/float(len(layer_sizes) - 1)

	    self.draw_nodes(ax, left, right, bottom, top, layer_sizes, 
	    				preactivations, values, layer_num, neuron_num, v_spacing, h_spacing, n_layers, curr_y)
	    
	    self.draw_edges(ax, left, right, bottom, top, layer_sizes, 
	    				preactivations, values, layer_num, neuron_num, v_spacing, h_spacing, n_layers, curr_y)
		


	def plot_error(self):
	    ax3.clear()
	    ax3.plot(range(len(self.Loss_arr)), self.Loss_arr)

	''' 
	Trains the given network with the training data passed.
	Calls, feed forward, then backprop, and finally updates gradients depending 
	on the solver chosen among the hyperparams
	'''
	# Function for animation
	def Animate(self, i):
		global img_count

		L_epoch=100000
		max_epoch, temp, self.beta1, self.beta2 = 3 , 0, 0.9, 0.999


		layer_val = list(self.sizes)
		layer_val.insert(0,self.input_layer_size)
		layer_val.append(self.output_layer_size)

		X = self.x_train
		y = self.y_train

		self.feed_forward(X[self.step * self.batch_size : (1+self.step) * self.batch_size, :], y)

		values = []
		values.append(X[self.step * self.batch_size, :])
		preactivations = []
		preactivations.append(X[self.step * self.batch_size, :])
		for p in range(len(self.h)):
			values.append(self.h[p][0])
			preactivations.append(self.a[p][0])
		
		# forward prop
		l_num = 1
		for l in layer_val[1:]:
			for num in range(0,l):
				self.draw_neural_net(fig.gca(), 0, 0.9, 0, 0.9, layer_val, preactivations, values, l_num, num)
				plt.pause(2)
				fig.savefig('Plots/img_' + str(img_count))
				img_count+=1
			l_num+=1

		curr_y = y[self.step * self.batch_size : (1+self.step) * self.batch_size, :]

		# loss computation
		self.draw_neural_net(fig.gca(), 0, 0.9, 0, 0.9, layer_val, preactivations, values, None, None, curr_y)
		plt.pause(2)
		fig.savefig('Plots/img_' + str(img_count))
		img_count+=1


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


		# backward prop
		self.draw_neural_net(fig.gca(), 0, 0.9, 0, 0.9, layer_val, preactivations, values, -(len(layer_val)-1), None, 'a')
		plt.pause(2)
		fig.savefig('Plots/img_' + str(img_count))
		img_count+=1
		l_num = -(len(layer_val)-2)
		for l in range(len(layer_val)-2):
			self.draw_neural_net(fig.gca(), 0, 0.9, 0, 0.9, layer_val, preactivations, values, l_num, None, 'w')
			plt.pause(2)
			fig.savefig('Plots/img_' + str(img_count))
			img_count+=1
			self.draw_neural_net(fig.gca(), 0, 0.9, 0, 0.9, layer_val, preactivations, values, l_num, None, 'h')
			plt.pause(2)
			fig.savefig('Plots/img_' + str(img_count))
			img_count+=1
			self.draw_neural_net(fig.gca(), 0, 0.9, 0, 0.9, layer_val, preactivations, values, l_num, None, 'a')
			plt.pause(2)
			fig.savefig('Plots/img_' + str(img_count))
			img_count+=1
			l_num+=1
		

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
  
  	for i in range(7):
  		nn.Animate(i)

  	# Start the animation and call training inside
 #  	ani = animation.FuncAnimation(fig, nn.Animate, frames=30, interval=1000)
	# ani.save('Neu_vis.mp4', writer = writer, savefig_kwargs={'facecolor':'black'})
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
    parser.add_argument('--loss', action="store", dest="loss", default="ce", type = str)
    parser.add_argument('--opt', action="store", dest="opt", default="adam", type = str)
    parser.add_argument('--batch_size', action="store", dest="batch_size", default=1, type = int)
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