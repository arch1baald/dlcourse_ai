import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    W2 = W * W
    loss = reg_strength * np.sum(W2)
    grad = 2 * reg_strength * W
    return loss, grad


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
    b = probs.shape[0]
    loss = np.mean(-np.log(probs[np.arange(b), target_index]))
    return loss


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
    if len(logits.shape) == 1:
        logits = logits.copy().reshape(1, -1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponents = np.exp(shifted)
    probs = exponents / np.sum(exponents, axis=1, keepdims=True)
    return probs


def softmax_with_cross_entropy(logits, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      logits, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      grad, np array same shape as predictions - gradient of predictions by loss value
    """
    if len(logits.shape) == 1:
        logits = logits.copy().reshape(1, -1)
    
    probs = softmax(logits)    
    loss = cross_entropy_loss(probs, target_index)
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
    return loss, grad


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.name = 'ReLU'
        self.x = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.x = X
        result = X.copy()
        result[result < 0] = 0
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        def heaviside(x):
            res = x.copy()
            res[res > 0] = 1
            res[res <= 0] = 0
            return res
        
        dx = d_out * heaviside(self.x)  # dL / dx
        return dx

    def params(self):
        # ReLU Doesn't have any parameters
        return {}
    
    def clear_grads(self):
        pass


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.name = 'FC'
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_input = np.dot(d_out, self.W.value.T)
        dW = np.dot(self.X.T, d_out)
        batch_size = d_out.shape[0]
        dB = np.dot(np.ones((1, batch_size)), d_out)
        self.W.grad += dW
        self.B.grad += dB
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
    
    def clear_grads(self):
        self.W.grad = np.zeros(self.W.value.shape)
        self.B.grad = np.zeros(self.B.value.shape)
