import numpy as np 
import time 
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from logsitic_sgd import load_data


class HiddenLayer(object):
	def __init__(self,rng,input,n_in,n_out,
		  		 weight=None,bias=None,activation=T.nnet.tanh):

		if weight is None:

			## tanh activation function
			W_values = np.asarray(
					np.uniform(
						low = -np.sqrt(6. / (n_in + n_out)),
						high = np.sqrt(6. / (n_in + n_out)),
						size = (n_in,n_out)
					),
					dtype = theano.config.floatX
				)
		if activation == T.nnet.sigmoid:
			W_values *=4

		Weight = theano.shared(value = W_values,borrow=True , name='weight')

		if bias is None:
			bias_value = np.zeros((n_out,),dtype = theano.config.floatX)
			bias = theano.shared(value=bias_value,name='bias',borrow=True)

		self.W = Weight
		self.bias = bias
		self.params = [self.W,self.b]
		self.input = input
		linear_output = T.dot(self.input,self.W) + self.b
		self.output = (linear_output if activation is None else activation(linear_output))

class dA(object):
	def __init__(self,random_rng,theano_rng,
				 input = None ,n_visible = 784,n_hdden = 500,
				 W = None, bhid = None,bvis = None):

		## bvis : bias of visable
		## bhid : bias of hidden


		if theano_rng is None:
			theano_rng = RandomStreams(random_rng.randint(2 ** 30))

		if W is None:
			W_values = np.asarray(
					random_rng.uniform(
						low = np.sqrt(6./ (n_visible + n_hidden )),
						high = np.sqrt(6. / (n_visible + n_hidden)),
						size = (n_visible,n_hidden)
					),
					dtype = theano.config.floatX
				)
		## sigmoid is 4 times 
		W_values *=4
		Weight = theano.shared(value = W_values , borrow = True , name='weight')

		if bhid is None:
			bhid = theano.shared(value = np.zeros((n_hidden,),dtype=theano.config.floatX),
								 borrow	= True,name='b')

		if bvis	is None:
			bvis = theano.shared(np.zeros((n_visible,),dtype = theano.config.floatX),
								 borrow= True)

		self.W = Weight
		self.b = bhid
		self.b_prime = bvis
		self.W_prime = self.W.T
		self.params  = [self.W,self.b,self.b_prime] ## save the parameter of model
		self.theano_rng = theano_rng
		self.input = input
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if input is None:
			self.x = T.dmatrix(name = 'input')
		else:
			self.x = self.input

	def get_corrupted_input(self,input,corruption_level):
		return self.theano_rng.binomial(size = input.shape,n=1,
										p = 1 - corruption_level,
										dtype = theano.config.floatX) * input

	def get_reconstructed_input(self,hidden):
		return T.nnet.sigmoid(T.dot(hidden,self.W_prime) + self.b_prime)

	def get_hidden_values(self,input):
		return T.nnet.sigmoid(T.dot(input,self.W) + self.b)

	def get_cost_updates(self,corruption_level,learning_rate):
		x = get_corrupted_input(self.x,corruption_level)
		y = get_hidden_values(x)
		z = get_reconstructed_input(y)

		L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
    
    	cost = T.mean(L)

    	## gradient descent
    	gparams = T.grad(cost,self.params)
    	updates = [(param, param - learning_rate * gparam)for param , gparam in zip(self.params , gparams)]

    	return (cost,updates)

## Stacking
class Sda(object):
	def __init__(self, numpy_rng, theano_rng = none,
				 n_ins=784,hidden_layers_sizes=[500,500],
				 n_outs=10,corruption_levels=[0.1,0.1]):

		self.sigmoid_layers  =[]
		self.dA_layers = []
		self.params = []
		self.n_layers = len(hidden_layers_sizes)  ## depth

		assert self.n_layers > 0

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		self.x = T.matrix('x')  
		self.y = T.ivector('y') ## one dimension


		for i in xrange(n_layers):
			if i ==0:
				input_size = n_ins
			else:
				input_size = hidden_layers_sizes[i-1]


			if i==0:
				layer_input = self.x
			else:
				layer_input = self.sigmoid_layers[i-1].output


			sigmoid_layer = HiddenLayer(rng = numpy_rng,input= layer_input , n_in = input_size,
										n_out = hidden_layers_sizes[i],activation=T.nnet.sigmoid)

			self.sigmoid_layers.append(sigmoid_layer)
			self.params.append(sigmoid_layer.params)

			## n_visible : total number
			## input : activation output
			dA_layer = dA(numpy_rng = numpy_rng,theano_rng=theano_rng,
						  input = layer_input,n_visible= input_size,
						  n_hidden = hidden_layers_sizes[i],
						  W = sigmoid_layer.W,bhid = sigmoid_layer.b)

			self.dA_layers.append(dA_layer)

		self.logLayer = LogsiticRegression(input  = sigmoid_layer[-1].output,
										   n_in = hidden_layers_sizes[-1], 
										   n_out = n_outs)	

		self.params.append(self.logLayer.params)
		self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
		self.errors = self.logLayer.error(self.y)


	def pretraining_function(self,train_x,batch_size):

		index =  T.lscalar('index') ## minibatch index
		corruption_level = T.scalar('corruption')
		learning_rate = T.scalar('1r')
		## beginning of a batch given index
		batch_begin = index * batch_size
		## end of a batch given index
		batch_end = batch_begin + batch_size

		pretrain_funs = []
		for da in self.dA_layers:
			## get the cost and the updates
			cost,updates = da.get_cost_updates(corruption_level,learning_rate)

			## compile the theano function
			fn = theano.function(
				inputs  = [
					index,
					theano.In(corruption_level,value=0.2),
					theano.In(learning_rate,value=0.1)
				],
				outputs = cost,
				updates = updates,
				givens = {
					self.x : train.x[batch_begin:batch_end]
				}
			)

			pretrain_funs.append(fn)

		return pretrain_funs


	def build_finetune_functions(self,datasets,batch_size,learning_rate):

		(train_x,train_y) = datasets[0]
		(valid_x,valid_y) = datasets[1]
		(test_x,test_y) = datasets[2]

		## calculate the number of minibatches 
		n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
		n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size	
		n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size

		index = T.lscalar('index') ## minibatch index
		gparams = T.grad(self.finetune_cost , self.params)  ## gradient descent
		updates = [(param, param - learning_rate * gparam) for param,gparam in zip(self.params,gparams)]

		## train model
		train_fn = theano.function(
				inputs = [index],
				outputs = self.finetune_cost,
				updates = updates,
				givens  = {
					self.x : train_x[index * batch_size : (index+1) * batch_size],
					self.y : train_y[index * batch_size : (index+1) * batch_size]
				},
				name = 'train'
			)

		## test model 
		test_fn = theano.function(
				inputs = [index],
				outputs = self.errors,
				givens = {
					self.x : test_x[index * batch_size : (index+1) * batch_size],
					self.y : test_y[index * batch_size : (index+1) * batch_size]
				},
				name='test'
			)

		## validation model
		valid_fn = theano.function(
				inputs = [index],
				outputs = self.errors,
				givens = {
					self.x : valid_x[index * batch_size : (index+1) * batch_size],
					self.y : valid_y[index * batch_size : (index+1) * batch_size]
				},
				name = 'valid'
			)


		def valid_score():
			return [valid_fn(i) for i in xrange(n_valid_batches)]

		def test_score():
			return [test_fn(i) for i in xrange(n_test_batches)]

		return train_fn,valid_score,test_score

if __name__ == "__main__":

	numpy_rng = np.random.RandomState(42)
	print 'building the model.....'

	sda = SdA(numpy_rng = numpy_rng, n_ins = 784 , hidden_layers_sizes = [1000,1000,1000],
			  n_outs = 10)




