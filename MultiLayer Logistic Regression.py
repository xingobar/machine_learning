import numpy as np
import time
import theano
from theano import tensor as T
from logistic_regression import LogisticRegression,load_data


class Hidden(object):
    '''
    For tanh activation function results obtained in [Xavier10] show that 
    the interval should be [-\sqrt{\frac{6}{fan_{in}+fan_{out}}},\sqrt{\frac{6}{fan_{in}+fan_{out}}}], 
    '''
    def __init__(self , rng, input, n_in , 
                 n_out , weight= None,  bias= None,activation=T.tanh):
        if weight is None:
            W_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6./(n_in + n_out)),
                    high = np.sqrt(6./(n_in + n_out)),
                    size = (n_in,n_out)
                ),
                dtype = theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_value *=4
            Weight = theano.shared(value=W_values,name='weight',borrow = True) # false is deepcopy

        if bias is None:
            bias_values = np.zeros((n_out,),dtype = theano.config.floatX)
            bias = theano.shared(value=bias_values,name='bias',borrow=True)

        self.W = Weight
        self.b = bias
        self.input = input
        linear_output = T.dot(input,self.W) + self.b
        self.output = (linear_output if activation is None else activation(linear_output))
        self.params = [self.W,self.b] # parameter of the model



class MLP(object):
    def __init__(self,random_stream,input,n_in,n_hidden,n_out = 10):
        
        self.hidden_layer = Hidden(rng = random_stream,
                                        input = input,
                                        n_in = n_in,
                                        n_out = n_hidden,
                                        activation = T.tanh)
        
        self.LogisticRegressionLayer = LogisticRegression(input = self.hidden_layer.output,
                                                          n_in = n_hidden,
                                                          n_out = 10) 
        
        ## compute l1 norm (sum) and squared l2 norm
        self.L1 = (
                    abs(self.hidden_layer.W).sum() + abs(self.LogisticRegressionLayer.W).sum()
                 )
        
        self.L2 = (
                     (self.hidden_layer.W **2).sum() + (self.LogisticRegressionLayer.W **2).sum()
                    )
        
        self.neg_loglikelihood = self.LogisticRegressionLayer.negative_log_likelihood
        self.error = self.LogisticRegressionLayer.error
        self.params = self.hidden_layer.params + self.LogisticRegressionLayer.params
        self.input = input


def test(learning_rate = 0.01,l1_reg=0.0,l2_reg=0.0001,
         n_epoch=1000,batch_size=20,hidden_units=500):
    
    dataset = load_data()
    train_x,train_y = dataset[0]
    validation_x,validation_y = dataset[1]
    test_x,test_y = dataset[2]
    
    ## compute the number of minibatches
    n_train_batches = train_x.get_value(borrow=True).shape[0] //  batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size
    n_validation_batches = validation_x.get_value(borrow=True).shape[0] // batch_size
    
    print 'building the model....'
    index = T.lscalar() ## index
    x = T.matrix('x')
    y = T.ivector('y') ## labels 
    random_state = np.random.RandomState(1234)
    
    classifier = MLP(random_stream = random_state,
                     input = x,
                     n_in = 28 * 28,
                     n_hidden = hidden_units,
                     n_out = 10)
   
    ## loss function (cost function) plus regularization (l1 norm and squared l2 norm)
    cost  = (
            classifier.neg_loglikelihood(y)  
            + l1_reg * classifier.L1 
            + l2_reg * classifier.L2
    )
    
    test_model = theano.function(
                inputs =[index],
                outputs = classifier.error(y),
                givens = {
                    x:test_x[index * batch_size : (index+1) * batch_size],
                    y:test_y[index * batch_size : (index+1) * batch_size]
        }
    )
    
    validation_model = theano.function(
                inputs = [index],
                outputs = classifier.error(y),
                givens ={
                    x:validation_x[index * batch_size : (index+1) * batch_size],
                    y:validation_y[index * batch_size : (index+1) * batch_size]
        }
    )
    ## gradient descent
    gparams = [T.grad(cost,params) for params in classifier.params]
    
    updates = [(params , params - learning_rate * gparams)  for params,gparams in zip(classifier.params,gparams)]
    
    train_model = theano.function(
                inputs = [index],
                outputs = cost,
                updates = updates,
                givens = {
                    x:train_x[index * batch_size : (index+1) * batch_size],
                    y:train_y[index * batch_size : (index+1) * batch_size]
        }
    )
    
    print 'complete the building model'
    print 'training the model....'
    
    ## early stopping
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches , patience //2) ## compute the validation per frequency
    best_validation_loss = np.inf
    best_iteration = 0.
    test_score = 0.
    start_time = time.time()
    epoch = 0
    looping = False
    
    while (epoch < n_epoch ) and (not looping):
        epoch +=1
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)
            iteration = (epoch -1) * n_train_batches + minibatch_index
            if (iteration +1) % validation_frequency ==0: ## per validation
                validation_loss = [validation_model(i) for i in xrange(n_validation_batches)] ## compute loss per validation
                validation_loss_mean = np.mean(validation_loss)
                print ' %i epoch %i/%i minibatch , validation error %f' %(epoch,
                                                                          minibatch_index+1,
                                                                          n_train_batches,
                                                                          validation_loss_mean * 100.)
                ## got the best validation score and we predict the test dataset
                if validation_loss_mean <  best_validation_loss:
                    if (validation_loss_mean < best_validation_loss * improvement_threshold):
                        patience = max(patience, iteration * patience_increase) 
                    ## save the best validation score and itearation
                    best_validation_loss = validation_loss_mean
                    best_iteration = iteration
        
        
                    ## predict the test set
                    test_score = [test_model(i) for i in xrange(n_test_batches)]
                    test_score_mean = np.mean(test_score)
                    
                    print ' %i epoch , %i/%i minibatch , test score %f ' %(epoch,
                                                                           minibatch_index +1,
                                                                           n_train_batches,
                                                                           test_score_mean * 100.)
            if patience <= iteration:
                looping = True
                break
    
    end_time = time.time()
    print 'complete the training model'
    print 'Best validation loss %f \n Best iteration %d \n Test Score %f' %(best_validation_loss * 100 , 
                                                                            best_iteration,
                                                                            test_score_mean * 100.)
    print 'Time is %0.2f' %((end_time - start_time) / 60)

if __name__ == "__main__":
	test()

