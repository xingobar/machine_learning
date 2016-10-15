# http://stackoverflow.com/questions/33416857/typeerror-cannot-convert-type-tensortypefloat64-vector-of-variable-subtenso
import theano
import numpy as np
import pickle
import time


def load_data():
    print 'Loading data....'
    with open('/Users/xingobar/Downloads/mnist.pkl', 'rb') as f:
        train_data,validation_data,test_data = pickle.load(f)
    
    def share_dataset(data,borrow=True):
        data_x,data_y = data
        shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x,shared_y
    
    train_x,train_y = share_dataset(train_data)
    validation_x,validation_y = share_dataset(validation_data)
    test_x,test_y = share_dataset(test_data)
    flatten = [(train_x,theano.tensor.cast(train_y.flatten(),'int32')),
               (validation_x,theano.tensor.cast(validation_y.flatten(),'int32')),
               (test_x,theano.tensor.cast(test_y.flatten(),'int32'))]
    return flatten


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



def sgd_optimization(learning_rate = 0.13,n_epoch = 1000, batch_size=600):
    dataset = load_data()
    train_x,train_y = dataset[0]
    validation_x,validation_y = dataset[1]
    test_x,test_y = dataset[2]
    n_train_batch = train_x.get_value(borrow=True).shape[0] / batch_size
    n_validation_batch  = validation_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batch = test_x.get_value(borrow=True).shape[0] / batch_size
    print 'number of training batches : ' , n_train_batch
    
    
    print 'building the model....'
    index = theano.tensor.lscalar() # index to a minibatch
    #learning_rate = 0.01
    x = theano.tensor.matrix('x') # data
    y =  theano.tensor.ivector('y') # labels , 1D vector
    classifier = LogisticRegression(input = x , n_in = 28*28,n_out = 10)
    cost  = classifier.negative_log_likelihood(y) # loss function

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
    givens = {
            x:validation_x[index * batch_size : (index+1) * batch_size],
            y:validation_y[index * batch_size : (index+1) * batch_size]
        }
    )
    # gradient descent
    g_w = theano.tensor.grad(cost=cost,wrt=classifier.W)
    g_b = theano.tensor.grad(cost=cost,wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_w),
               (classifier.b,classifier.b - learning_rate * g_b)]

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
    
    print 'Start training the model.....'
    ## early stopping
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batch,patience // 2)
    best_validation_loss = np.inf
    test_score = 0 
    start_time = time.time()
    epoch = 0 
    looping = False
    
    while (epoch < n_epoch) and (not looping):
        epoch +=1
        for minibatch_index in xrange(n_train_batch):
            minibatch_cost = train_model(minibatch_index) ## output cost
            n_iteration = (epoch - 1  ) * n_train_batch + minibatch_index
            if (n_iteration + 1)  % validation_frequency ==0 : ## compute the validation losses per validation
                validation_losses = [validation_model(i) for i in xrange(n_validation_batch)] ## output errors
                this_validation_losses = np.mean(validation_losses)
                print 'epoch %i , minibatch %i/%i , validation error %f ' %(epoch,
                                                                            minibatch_index+1,
                                                                            n_train_batch,
                                                                            this_validation_losses * 100)
    
                if this_validation_losses < best_validation_loss:
                    if this_validation_losses < best_validation_loss * improvement_threshold:
                        patience = max(patience , n_iteration * patience_increase)
                    best_validation_loss = this_validation_losses
                
                    test_losses = [test_model(i) for i in xrange(n_test_batch)] ## output errors
                    test_score = np.mean(test_losses)
                    print 'epoch %i, minibatch %i/%i , test score %f ' %(epoch,
                                                                      minibatch_index+1,
                                                                      n_train_batch,
                                                                      test_score * 100)
                    
                    ## save the best model 
                    with open('logistic_regression_best_model.pkl','wb') as f:
                        pickle.dump(classifier,f)
                        
            if patience <= n_iteration:
                looping = True
                break
    end_time = time.time()          
    print 'Complete the trainig the model.....'
    print 'Time is %0.2f' %((end_time - start_time) / 60)
    print 'best validation score of %f with test score %f' % (best_validation_loss * 100, test_score*100)



def predict():
    classifier = pickle.load(open('logistic_regression_best_model.pkl','rb'))
    predicited_model = theano.function(
    inputs =[classifier.input],
    outputs = classifier.y_pred
    )
    
    dataset = load_data()
    test_x,test_y = dataset[2]
    test_x = test_x.get_value()
    y_pred = predicited_model(test_x[:10])
    print y_pred

if __name__ =='__main__':
    sgd_optimization()
    predict()
