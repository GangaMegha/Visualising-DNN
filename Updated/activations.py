import numpy as np 

class active():
	def sigmoid(self,x):
		#Sigmoid activation
		#Implemented interms  of tanh for increased stability
		return 1.0/(1 + np.exp(-x))

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