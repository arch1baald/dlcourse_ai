import numpy as np


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
    W2 = W * W
    loss = reg_strength * np.sum(W2)
    grad = 2 * reg_strength * W
    return loss, grad


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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.name = 'ReLU'
        self.x = None

    def forward(self, X):
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
        def heaviside(x):
            res = x.copy()
            res[res > 0] = 1
            res[res <= 0] = 0
            return res
        
        dx = d_out * heaviside(self.x)  # dL / dx
        return dx

    def params(self):
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


class ConvolutionalLayer:
    def __init__(self, in_channels, n_filters,
                 filter_size, padding, stride=1):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        n_filters, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''
        self.X = None
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.W = Param(np.random.randn(
            # Здесь n_filters в конце, а не в начале!
            filter_size, filter_size, in_channels, n_filters
        ))
        self.B = Param(np.zeros(n_filters))
        self.padding = padding
        self.stride = stride

    def forward(self, X):
        self.X = X.copy()
        batch_size, input_height, input_width, in_channels = X.shape
        out_height, out_width = self.get_output_shape(input_height, input_width)
        
        # В итоге z будет иметь другую размерность, см. конец ф-ии!
        z = np.zeros((batch_size, self.n_filters, out_height, out_width))
        for y in range(out_height):
            for x in range(out_width):
                x_start = x * self.stride
                x_end = x_start + self.filter_size
                y_start = y * self.stride
                y_end = y_start + self.filter_size
                # Не забыть про reshape!
                w = self.W.value.reshape(self.n_filters, self.filter_size, self.filter_size, in_channels)
                filtered = w * X[:, np.newaxis, y_start:y_end, x_start:x_end, :]
                z[:, :, y, x] = np.sum(filtered, axis=(2, 3, 4))

        # n_filters в конце, а не после batch_size, чтобы работало суммирование с self.B
        z = z.reshape(batch_size, out_height, out_width, self.n_filters)
        z += self.B.value
        return z

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        if self.X is None:
            raise Exception('Backward before forward')

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, n_filters = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        
        d_input = np.zeros_like(self.X).astype(np.float)
        # Не проебать момент, что в forward reshape W!
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                x_start = x * self.stride
                x_end = x_start + self.filter_size
                y_start = y * self.stride
                y_end = y_start + self.filter_size
#         https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L103
# Возможно стоит задить хуй на сведение к FC и сделать как просто в Conv ^
                X_local = self.X[:, y_start:y_end, x_start:x_end, :]
                original_local_shape = X_local.shape
                hwc = self.filter_size * self.filter_size * self.in_channels
                X_local = X_local.reshape(batch_size, hwc)
                w = self.W.value.reshape(hwc, self.n_filters)
                b = self.B.value.reshape(1, self.n_filters)
                fc = FullyConnectedLayer(*w.shape)
                fc.X = X_local
                fc.W.value = w
                fc.B.value = b
                d_local_out = d_out[:, y, x, :]
                
                d_local_input = fc.backward(d_local_out)
                # Полный проеб
                d_local_input = d_local_input.reshape(*original_local_shape)
                d_input[:, y_start:y_end, x_start:x_end, :] += d_local_input
#                 # Где-то проеб с порядком индексов в 2х2, в 3х3 вообще не работает
                w_grad = fc.W.grad.reshape(*self.W.grad.shape)
#                 w_grad = fc.W.grad.reshape(self.n_filters, self.filter_size, self.filter_size, self.in_channels)
#                 w_grad = fc.W.grad.reshape(self.filter_size, self.filter_size, self.in_channels, self.n_filters)
                self.W.grad += w_grad
                # Все окей
                b_grad = fc.B.grad.reshape(*self.B.grad.shape)
                self.B.grad += b_grad
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    def get_output_shape(self, input_height, input_width):
        out_height = 1 + (input_height - self.filter_size + 2 * self.padding) // self.stride
        out_width = 1 + (input_width - self.filter_size + 2 * self.padding) // self.stride
        return out_height, out_width


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
