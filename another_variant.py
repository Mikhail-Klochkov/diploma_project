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


class function(tf.Module):
    def __init__(self, x_numpy):
        if(x_numpy.shape.__len__() != 1):
            assert (1 != 1) ('shape is not signature : (n, )!')
        else:
            self._tensor = tf.Variable(x_numpy)

    def __call__(self, some_parameter = 0):
        self._another = tf.convert_to_tensor(rosen(self._tensor.numpy()))
        return self._another

    def __repr__(self):
        return (self._tensor.__repr__(), self._another.__repr__()).__str__()

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

x_tens_ = tf.constant(np.full((5,), 2.))

x_another = tf.constant(np.full((5,), 1.5))

x_tens_.shape.__len__()
foo_ = function(x_tens_)
param_ = 3
input_(foo_.__call__(param_))

if(x_another.shape == x_tens_.shape):
    print("hell")

else:
    print("not hell")

foo_
foo_._change(np.full((5, ), 3.))
foo_
foo_.__call__(param_)

function_ = foo_._wrapper()

function_(x_another)
from scipy.optimize import minimize
x_0 = np.array([1.3 , 0.7 , 0.8, 3.0, 2.0])
res_ = minimize(function_, x_0, method = 'nelder-mead', options = {'xtol': 1e-8, 'disp': True})

print(res_.x)
