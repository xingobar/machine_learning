import numpy as np 
import time 
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from logistic_sgd import load_data

class LogisticRegression(object):
    def __init__(self,input,n_in,n_out):
        self.W = theano.shared(np.zeros((n_in,n_out),dtype=theano.config.floatX),name='weight',borrow=True)
        self.b = theano.shared(np.zeros((n_out),dtype=theano.config.floatX),name='bias',borrow=True)
        self.y_prob_given_x  =  theano.tensor.nnet.softmax(theano.tensor.dot(input,self.W) + self.b)
        self.y_pred = theano.tensor.argmax(self.y_prob_given_x,axis=1) 
        self.params = [self.W,self.b]
        self.input = input
    
    def negative_log_likelihood(self,y):
        return -theano.tensor.mean(theano.tensor.log(self.y_prob_given_x)[theano.tensor.arange(y.shape[0]),y])
    
    def error(self,y):
        if y.dtype.startswith('int'):
            return theano.tensor.mean(theano.tensor.neq(self.y_pred,y))

class HiddenLayer(object):
	def __init__(self,rng,input,n_in,n_out,
		  		 weight=None,bias=None,activation=T.tanh):

		if weight is None:

			## tanh activation function
			W_values = np.asarray(
					rng.uniform(
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
		self.b = bias
		self.params = [self.W,self.b]
		self.input = input
		linear_output = T.dot(self.input,self.W) + self.b
		self.output = (linear_output if activation is None else activation(linear_output))

class dA(object):
    
    def __init__(self,numpy_rng,theano_rng,input=None,n_visible=784,n_hidden=500,W=None,bhid=None,bvis=None):
        
        ## bhid : bias values for hidden units
        ## bvis : bias values for visible units
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        
        if not W:
            inital_W = np.asarray(
                            numpy_rng.uniform(
                                low = -4 * np.sqrt(6. / (n_visible + n_hidden)),
                                high = 4 * np.sqrt(6. / (n_visible + n_hidden)),
                                size = (n_visible,n_hidden)
                            ),dtype = theano.config.floatX
                        )
            self.W = theano.shared(value = inital_W ,name='W',borrow= True)
        else:
        	#inital_W = W
        	self.W = W
        #Weight = theano.shared(value = inital_W, name = 'W',borrow = True)
        
        if not bvis:
            bvis = theano.shared(
                        value = np.zeros((n_visible,) , dtype = theano.config.floatX),
                        borrow = True
                    )
        
        if not bhid:
            bhid = theano.shared(
                        value = np.zeros((n_hidden,), dtype = theano.config.floatX),
                        borrow = True,
                        name = 'b'
                    )
        
        #self.W = Weight
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        
        self.params = [self.W , self.b , self.b_prime] ## save the parameter of model
    
    
    def get_corrupted_input(self,input,corruption_level):
        
        return self.theano_rng.binomial(size = input.shape, n =1 ,
                                   p = 1 - corruption_level,
                                   dtype = theano.config.floatX) * input
    
    def get_hidden_values(self,input):
        
        return T.nnet.sigmoid(T.dot(input,self.W) + self.b)
    
    def get_reconstructed_values(self,hidden):
        
        return T.nnet.sigmoid(T.dot(hidden,self.W_prime) + self.b_prime)
    
    def get_cost_updates(self,corruption_level, learning_rate):
        
        x = self.get_corrupted_input(self.x,corruption_level)
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_values(y)
        
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
    
        cost = T.mean(L)
        
        ## gradient descent
        gparams = T.grad(cost,self.params)
        ## generate updates
        updates = [(param, param - learning_rate * gparam) for param,gparam in zip(self.params , gparams)]
        
        return (cost,updates)


## Stacking
class SdA(object):
	def __init__(self, numpy_rng, theano_rng = None,
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


		for i in xrange(self.n_layers):
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
			self.params.extend(sigmoid_layer.params)

			## n_visible : total number
			## input : activation output
			dA_layer = dA(numpy_rng = numpy_rng,theano_rng=theano_rng,
						  input = layer_input,n_visible= input_size,
						  n_hidden = hidden_layers_sizes[i],
						  W = sigmoid_layer.W,bhid = sigmoid_layer.b)

			self.dA_layers.append(dA_layer)

		self.logLayer = LogisticRegression(input  = self.sigmoid_layers[-1].output,
										   n_in = hidden_layers_sizes[-1], 
										   n_out = n_outs)	

		self.params.extend(self.logLayer.params)
		self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
		self.errors = self.logLayer.error(self.y)


	def pretraining_function(self,train_x,batch_size):

		index =  T.lscalar('index') ## minibatch index
		corruption_level = T.scalar('corruption')
		learning_rate = T.scalar('lr')
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
					self.x : train_x[batch_begin:batch_end]
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


def test(finetune_lr=0.1,pretraining_epochs = 15,pretrain_lr=0.001,
		 training_epochs = 1000 , batch_size = 1):

	datasets = load_data()
	train_x,train_y = datasets[0]
	valid_x,valid_y = datasets[1]
	test_x,test_y = datasets[2]

	n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size

	numpy_rng = np.random.RandomState(42)
	print 'building the model.....'

	sda = SdA(numpy_rng = numpy_rng, n_ins = 784 , hidden_layers_sizes = [1000,1000,1000],
			  n_outs = 10)


	pretraining_fns = sda.pretraining_function(train_x = train_x,
											   batch_size=batch_size)

	start_time = time.time()
	corruption_level = [.1,.2,.3]
	for i in xrange(sda.n_layers):
		for epoch in xrange(pretraining_epochs):
			result	= []
			for minibatch_index in xrange(n_train_batches):
				result.append(pretraining_fns[i](index = minibatch_index,
										  corruption = corruption_level[i],
										  lr= pretrain_lr))
			print 'layer %i , epoch %d, cost %f' %(i,epoch,np.mean(result))

	end_time = time.time()
	print "Time is %0.2f " %((end_time - start_time) / 60)

	print 'getting the finetuning function ...'
	train_fn,valid_model,test_model = sda.build_finetune_functions(
			datasets = datasets,
			batch_size = batch_size,
			learning_rate = finetune_lr
		)

	print 'finetuning the model ....'
	## early stopping
	patience = 10 * n_train_batches
	patience_increase = 2.
	improvement_threshold = 0.995
	validation_frequency = min(n_train_batches, patience //2)

	best_validation = np.inf
	epoch = 0
	test_score =0.
	start_time = time.time()
	looping = False

	while ( epoch < training_epochs) and (not looping):
		epoch +=1
		for minibatch_index in xrange(n_train_batches):
			iteration = (epoch -1 ) * n_train_batches + minibatch_index

			if (iteration + 1) % validation_frequency == 0:
				validation_lossess = valid_model()
				validation_mean = np.mean(validation_lossess)
				print 'epoch %i minibatch %i/%i, validation_error %f' % (epoch, 
																	     minibatch_index + 1 , 
																	     n_train_batches , 
																	     validation_mean * 100.) 


				if validation_mean < best_validation:
					if validation_mean < best_validation * improvement_threshold:
						patience = max(patience , iteration * patience_increase)

				## save best validation model
				best_validation = validation_mean
				best_iter = iteration
				
				## test 
				test_lossess = test_model()
				test_mean = np.mean(test_lossess)
				print 'epoch %i minibatch %i/%i, test score %f' % (epoch,
																   minibatch_index + 1,
																   n_train_batches,
																   test_mean * 100.)


		if patience <= iteration:
			looping = True
			break

	end_tiem = time.time()
	print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation * 100., best_iter + 1, test_mean * 100.)
    )

if __name__ == "__main__":
	test()

