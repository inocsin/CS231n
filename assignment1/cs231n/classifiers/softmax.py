import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_class=W.shape[1]
  num_train=X.shape[0]

  for i in xrange(num_train):
    f_i=np.dot(X[i,:],W)

    log_c=np.max(f_i)
    f_i-=log_c

    sum_i=0.0
    for f_i_j in f_i:
      sum_i+=np.exp(f_i_j)
    loss+=-f_i[y[i]]+np.log(sum_i)

    for j in xrange(num_class):
      p=np.exp(f_i[j])/sum_i
      dW[:,j]+=(p-(j==y[i]))*X[i,:].T

  loss/=num_train
  dW/=num_train

  loss+=0.5*reg*np.sum(W*W)
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_class = W.shape[1]
  num_train=X.shape[0]

  f=np.dot(X,W)
  f-=np.max(f,axis=1,keepdims=True)
  f_correct=f[range(num_train),y]
  loss=-np.mean(np.exp(f_correct)/np.sum(np.exp(f),axis=1,keepdims=True))


  p=np.exp(f)/np.sum(np.exp(f),axis=1,keepdims=True)
  ind=np.zeros(p.shape)
  ind[range(num_train),y]=1
  dW=np.dot(X.T,(p-ind))
  dW/=num_train

  loss+=0.5*reg*np.sum(W*W)
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

