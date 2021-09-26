import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)
        # Если добавлять ReLU перед softmax, то сеть плохо обучается
        # и не может достичь нужного качества
#         self.relu2 = ReLULayer()
        self.nn = [
            self.fc1,
            self.relu1,
            self.fc2,
#             self.relu2,
        ]
        
    def forward(self, X):
        x = None
        for i, layer in enumerate(self.nn):
            if i == 0:
                x = layer.forward(X)
            else:
                x = layer.forward(x)
        return x

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for layer in self.nn:
            layer.clear_grads()
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        x = self.forward(X)
        loss, grad = softmax_with_cross_entropy(x, y)
        
        for i, layer in reversed(list(enumerate(self.nn))):
            layer_params = layer.params()
            if layer_params:
                W = layer_params['W'].value
                B = layer_params['B'].value
                regloss, reggrad = l2_regularization(W, self.reg)
                layer_params['W'].grad += reggrad
                loss += regloss
                regloss, reggrad = l2_regularization(B, self.reg)
                layer_params['B'].grad += reggrad
                loss += regloss
            grad = layer.backward(grad)
        return loss
    
    def predict_proba(self, X):
        x = self.forward(X)
        probs = softmax(x)
        return probs

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        probs = self.predict_proba(X)
        pred = np.argmax(probs, axis=1)
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = dict()
#         for name, layer in zip(['fc1', 'relu1', 'fc2', 'relu2'], self.nn):
        for i, layer in enumerate(self.nn):
            name = f'{layer.name}{i}'
            w_key = f'{name}_w'
            b_key = f'{name}_b'
            if layer.params():
                result[w_key] = layer.params()['W']
                result[b_key] = layer.params()['B']
        return result
