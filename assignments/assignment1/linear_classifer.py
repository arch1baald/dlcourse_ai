import numpy as np


def softmax(logits):
    '''
    Computes probabilities from logits

    Arguments:
      logits, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as logits - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
#     shifted_logits = logits - np.max(logits, axis=1)
#     probs = np.exp(shifted_logits) / np.sum(np.exp(shifted_logits))

#   Example for better understanding of np.newaxis
#     asdf = np.array(
#         [[ 2., -1., -1.,  1.],
#         [ 0.,  1.,  1.,  1.],
#         [ 1.,  2., -1.,  2.]]
#     )
#     rmax = np.max(asdf, axis=1)
#     rmax = rmax[:, np.newaxis]
#     asdf - rmax

#     Еще проще можно использовать keepdims=True
#     https://cs231n.github.io/neural-networks-case-study/#grad
#     asdf = np.array(
#         [[ 2., -1., -1.,  1.],
#         [ 0.,  1.,  1.,  1.],
#         [ 1.,  2., -1.,  2.]]
#     )
#     rmax = np.max(asdf, axis=1, keepdims=True)
#     asdf - rmax

#     if len(logits.shape) == 1:
#         logits = logits.copy().reshape(1, -1)

#     row_max = np.max(logits, axis=1)
#     row_max = row_max[:, np.newaxis]
#     shifted_logits = logits - row_max
#     exponents = np.exp(shifted_logits)
#     denum = np.sum(exponents, axis=1)
#     denum = denum[:, np.newaxis]
#     probs = exponents / denum

    if len(logits.shape) == 1:
        logits = logits.copy().reshape(1, -1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponents = np.exp(shifted)
    probs = exponents / np.sum(exponents, axis=1, keepdims=True)
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
#     loss = -np.log(probs[target_index])
    b = probs.shape[0]
    loss = np.mean(-np.log(probs[np.arange(b), target_index]))
    return loss


def softmax_with_cross_entropy(logits, target_index):
    '''
    Computes softmax and cross-entropy loss for last linear + softmax layer,
    including the gradient

    Arguments:
      logits, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      grad, np array same shape as predictions - gradient of logits by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    # https://deepnotes.io/softmax-crossentropy
    # https://cs231n.github.io/neural-networks-case-study/#grad
    if len(logits.shape) == 1:
        logits = logits.copy().reshape(1, -1)
    
    probs = softmax(logits)    
    loss = cross_entropy_loss(probs, target_index)
#     target = np.zeros_like(logits)
#     target[:, target_index] = 1
    b = probs.shape[0]
    k = probs.shape[1]
    target = np.zeros((b, k), dtype=np.float)
    if isinstance(target_index, int):
        target[np.arange(b), target_index] = 1.
        grad = probs - target
        grad = grad.reshape(k, )
    else:
        target[np.arange(b), target_index.flatten()] = 1.
        grad = (probs - target) / b
        
#     print('probs:')
#     display(probs)
#     print('target_represented:')
#     display(target)
    
#     print('loss', loss)
#     print('grad\n', grad)
#     print()
    return loss, grad


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    
    # Поэлементное умножение, т.е. возведение в квадрат каждого w_ij
    W2 = W * W
    loss = reg_strength * np.sum(W2)
    grad = 2 * reg_strength * W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    logits = np.dot(X, W)
    
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss, grad = softmax_with_cross_entropy(logits, target_index)
    
    # https://cs231n.github.io/neural-networks-case-study/#together
    # https://cs231n.github.io/optimization-2/
    # http://cs231n.stanford.edu/vecDerivs.pdf
    # TODO: Честно разобраться почему именно так перемножаются
    # Tip: Якобиан (градиент) весов всегода совпадает по размерности с W
    dW = np.dot(X.T, grad)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            loss = 0
            for idx in batches_indices:
                X_batch = X[idx, ]
                logloss, dL_W = linear_softmax(X, self.W, y)
                regloss, dR_W = l2_regularization(self.W, reg)
                loss += logloss + regloss
                grad = dL_W + dR_W
                self.W -= learning_rate * grad
            # end
            print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)
            eps = learning_rate / 10
            if epoch > 0:
                if np.abs(loss_history[-1] - loss_history[-2]) < eps:
                    break
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        logits = np.dot(X, self.W)
        probs = softmax(logits)
        y_pred = np.argmax(probs, axis=1)
        return y_pred
