import tensorflow.compat.v2 as tf
from scipy.optimize import minimize
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

@tf.function
def input_tens(tens):
    tf.print(tens)

class function(tf.Module):
 
    def __init__(self, x_numpy, functional_obj):
        if(x_numpy.shape.__len__() != 1):
            assert (1 != 1) ('shape is not signature : (n, )!')
        else:
            print(" __init__: ", x_numpy, functional_obj, x_numpy.shape, '\n end __init__')
            self._tensor = tf.Variable(x_numpy)
            self._times_call_change = 0
            self._functional_obj = functional_obj 
            self.stack_arg = [] 
    def __call__(self, some_parameter = 0): 
        return self._functional_obj(self._tensor) 

    def __repr__(self):
        return (self._tensor.__repr__(), self._functional_obj.__repr__())

    def _change(self, new_vector):
        self._times_call_change += 1
        @tf.function
        def assign_(x_new):
            x_new_tensor = tf.convert_to_tensor(x_new)
            if(x_new_tensor.shape != self._tensor.shape):
                assert (1 != 1) ("shape of tensor is not a equal!")
            else:
                self._tensor.assign(x_new_tensor)
        if(self._times_call_change % 4 == 0):
            print('current value of x: {!s} \n current of Function: {!s} \n'.format(self._tensor.numpy(), 
            							self.__call__().numpy()))
            self.stack_arg.append(self._tensor.numpy())

        assign_(new_vector)

    def _wrapper(self): 
        def _inner(vars_x) -> np.float32:
            if(vars_x.shape[0] != self._tensor.shape[0]):
                assert (1 != 1) ('Error with different size!: ')
                return None
            else:
                self._change(vars_x) 
                return (self.__call__(some_parameter = 0)).numpy()
        return _inner

    def compute_gradient(self):
                with tf.GradientTape() as g:
                    g.watch((self._tensor))
                    f_a = self._functional_obj(self._tensor)
                    grad_ = g.gradient(f_a, self._tensor)

                return grad_

    def wrapper_grad(self):
        def inner_grad(vars_x, *args) -> float: 
            if(vars_x.shape[0] != self._tensor.shape[0]):
                assert (1 != 1) ('Error with different size!: ')
                return None
            else:
                self._change(vars_x)
                return self.compute_gradient().numpy()

        return inner_grad

    def compute_hessian(self):
        with tf.GradientTape(persistent = True) as g:
            g.watch(self._tensor)
            f_a = self._functional_obj(self._tensor)
            grad_ = g.gradient(f_a, self._tensor)
            list_ = []
            for iter_ in range(self._tensor.shape[0]):
                grad_grad = g.gradient(grad_[iter_], self._tensor)
                list_.append(grad_grad)

            matrix = tf.convert_to_tensor(list_)
        return matrix


    def wrapper_hessian(self):
        def inner_hessian(vars_x, *args) -> float:
            if(vars_x.shape[0] != self._tensor.shape[0]):
                assert (1 != 1) ('Error with different size!: ')
                return None
            else:
                self._change(vars_x)
                return self.compute_hessian().numpy()

        return inner_hessian
