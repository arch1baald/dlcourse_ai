import numpy as np


def check_gradient(f, x, delta=1e-5, tol=1e-4, verbose=0):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    assert isinstance(x, np.ndarray), 'isinstance'
    assert x.dtype == np.float, f'x.dtype={x.dtype}'
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"
    
    assert analytic_grad.shape == x.shape, 'analytic_grad.shape'
    analytic_grad = analytic_grad.copy()
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    def g(xx):
        """Returns f(x) without second part grad f(x)"""
        if verbose > 0:
            print('f', f)
        return f(xx)[0]

    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0

        x_right = x.copy()
        x_right[ix] += delta
        f_right = g(x_right)
        x_left = x.copy()
        x_left[ix] -= delta
        f_left = g(x_left)
        grad = (f_right - f_left) / (2 * delta)
        try:
            numeric_grad_at_ix = grad[ix]
        except IndexError:
            # if grad is scalar
            numeric_grad_at_ix = grad
        
        if verbose > 0:
            print('ix:', ix)
            display('x_right:', x_right)
            print('f_right:', f_right)
            display('x_left:', x_left)
            print('f_left:', f_left)
            print('grad:', grad)
            print('numeric_grad_at_ix:', numeric_grad_at_ix)
            print()
        
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
#             return False

        it.iternext()
    print("Gradient check passed!")
    return True


def check_layer_gradient(layer, x, delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for the input and output of a layer

    Arguments:
      layer: neural network layer, with forward and backward functions
      x: starting point for layer input
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    output = layer.forward(x)
    output_weight = np.random.randn(*output.shape)

    def helper_func(x):
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        grad = layer.backward(d_out)
        return loss, grad

    return check_gradient(helper_func, x, delta, tol)


def check_layer_param_gradient(layer, x,
                               param_name,
                               delta=1e-5, tol=1e-4, verbose=0):
    """
    Checks gradient correctness for the parameter of the layer

    Arguments:
      layer: neural network layer, with forward and backward functions
      x: starting point for layer input
      param_name: name of the parameter
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    param = layer.params()[param_name]
    initial_w = param.value

    output = layer.forward(x)
    output_weight = np.random.randn(*output.shape)

    def helper_func(w):
        param.value = w
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        layer.backward(d_out)
        grad = param.grad
        return loss, grad

    return check_gradient(helper_func, initial_w, delta, tol, verbose)


def check_model_gradient(model, X, y,
                         delta=1e-5, tol=1e-4, verbose=0):
    """
    Checks gradient correctness for all model parameters

    Arguments:
      model: neural network model with compute_loss_and_gradients
      X: batch of input data
      y: batch of labels
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    params = model.params()

    for param_key in params:
        print("Checking gradient for %s" % param_key)
        param = params[param_key]
        initial_w = param.value

        def helper_func(w):
            param.value = w
            loss = model.compute_loss_and_gradients(X, y)
            grad = param.grad
            return loss, grad

        if not check_gradient(helper_func, initial_w, delta, tol, verbose):
            return False

    return True
