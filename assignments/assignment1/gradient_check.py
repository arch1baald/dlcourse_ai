import numpy as np


def check_gradient(f, x, delta=1e-5, tol=1e-4):
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
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"
    
    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    def g(xx):
        """Returns f(x) without second part grad f(x)"""
        return f(xx)[0]
    
#     print('\n\n --==Numerical Gradient==--')
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0
#         print('\n\nix:', ix)
#         print('analytic_grad_at_ix:', analytic_grad_at_ix)
        
        # TODO compute value of numeric gradient of f to idx
        x_right = x.copy()
        x_right[ix] += delta
        x_left = x.copy()
        x_left[ix] -= delta
        numeric_grad_at_ix = (g(x_right) - g(x_left)) / (2 * delta)
        # Note: we dont need 2 copies of x, could be optimized
#         display('x_right:', x_right)
#         display('x_left:', x_left)
#         print('g(x_right):', g(x_right))
#         print('g(x_left):', g(x_left))
#         print('numeric_grad_at_ix:', numeric_grad_at_ix)
#         print()
        
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

        

        
