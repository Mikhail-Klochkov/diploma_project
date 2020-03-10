import tensorflow.compat.v2 as tf
from scipy.optimize import minimize
import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

np.sum(np.array([a for a in range(1, 10,1)]))
x_ = np.arange(start = -1, stop = 0, step = 0.1)
np.sum(x_[x_.__len__() // 4 :x_.__len__()//2])
x_range_ = np.arange(4)
x_range_[:-1] # по последнему не смотрят
#x_range_[1:] # по первому игнор



@tf.function
def input_tens(tens):
    tf.print(tens)

def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)

x_another = [x_range_,-x_range_]
print(x_another)
np.sum(x_another)

x = tf.convert_to_tensor(np.full((2,1), 1))

@tf.function
def input_(x_tens):
    tf.print(x_tens)

input_(x)

x_ = np.full((3,), 2)
rosen(x_)

tens_ = tf.convert_to_tensor(rosen(x_))
input_(tens_)
tens_.__repr__()

def rosen_tf(x_tens):
    return tf.reduce_sum(100.0 * (x_tens[1:] - x_tens[:-1] ** 2) ** 2 + (1 - x_tens[:-1]) ** 2)

def diff_(x_tens):
    return tf.reduce_sum(x_tens) * 3.0

class function(tf.Module):

    def compute_gradient_optional(self, functional_obj):
        with tf.GradientTape() as g:
            g.watch(self._tensor)
            res = functional_obj(self._tensor)
            grad_ = g.gradient(res, self._tensor)

        return grad_

    def __init__(self, x_numpy):

        assert( (x_numpy.shape.__len__() == 1), \
               ('We tae a x_numpy shape not (ndims, ) ! \n shape tensor is : {!s}'.format(x_numpy.shape.__len__())))

        if(x_numpy.shape.__len__() != 1):
            assert (1 != 1) ('shape is not signature : (n, )!')
        else:
            self._tensor = tf.Variable(x_numpy)
            self._dims = x_numpy.shape[0]

    def _call_f(self, function = rosen):
       return tf.converct_to_tensor(function(self._tensor)) ## При условии что ее вообще можно преобразовать в тензор

    def __call__(self, some_parameter = 0): # define a default parameter rosen function
        self._another = tf.convert_to_tensor(rosen(self._tensor.numpy()))
        return self._another  ## we can define call for another functional pbject

    def __repr__(self):
        return (self._tensor.__repr__(), self._another.__repr__())

    def __str__(self):

        return "Dimensional is {!s} \n Vector x is : {!s} \n Value of functional is: {!s} \n".format(
                                      self._dims, self._tensor.numpy(), self.__call__(some_parameter = 0).numpy())

    def _change(self, new_vector):
        @tf.function
        def assign_(x_new):
            x_new_tensor = tf.convert_to_tensor(x_new)
            if(x_new_tensor.shape != self._tensor.shape):
                assert (1 != 1) ("shape of tensor is not a equal!")
            else:
                self._tensor.assign(x_new_tensor)

        assign_(new_vector)

    def _wrapper(self):
        def _inner(vars_x) -> np.float32:
            if(vars_x.shape[0] != self._tensor.shape[0]):
                assert (1 != 1) ('Error with different size!: ')
                return None
            else:
                self._change(vars_x) # we change aor vector of x
                return (self.__call__(some_parameter = 0)).numpy()

        return _inner

    def compute_gradient(self):
                with tf.GradientTape() as g:
                    g.watch((self._tensor))
                    f_a = tf.reduce_sum(100.0 * (self._tensor[1:] - self._tensor[:-1] ** 2) ** 2 + (1 - self._tensor[:-1]) ** 2)
                    grad_ = g.gradient(f_a, self._tensor)

                return grad_

    def wrapper_grad(self):
        def inner_grad(vars_x, *args) -> float: # should be a vector of shape np.float
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
            f_a = tf.reduce_sum(100.0 * (self._tensor[1:] - self._tensor[:-1] ** 2) ** 2 + (1 - self._tensor[:-1]) ** 2)
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


x_tens_ = tf.constant(np.full((5,), 2.))
x_another = tf.constant(np.full((5,), 1.5))

foo_ = function(x_tens_)
print("proba: ", foo_.compute_gradient_optional(diff_))
param_ = 3
input_(foo_.__call__(param_))

foo_._change(np.full((5, ), 2.))
function_ = foo_._wrapper()
funct_grad = foo_.wrapper_grad()

input_tens(foo_.compute_gradient())

funct_hessian = foo_.wrapper_hessian()


#print(funct_grad(np.full((5,), 0.0)))

from scipy.optimize import minimize
x_0 = np.array([1.3 , 0.7 , 0.8, 3.0, 2.0])
res_ = minimize(function_, x_0, method = 'nelder-mead', options = {'xtol': 1e-8, 'disp': True})
print(res_.x)
res_an_ = minimize(function_, x_0, method = 'BFGS', jac = funct_grad, options = {'disp': False})

print(res_an_)

res_another = minimize(function_, x_0, method = 'Newton-CG', jac = funct_grad,hess = funct_hessian,options = {'xtol' : 1e-8, 'disp': True})
print(res_another)

