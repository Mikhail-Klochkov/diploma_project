
import tensorflow.compat.v2 as tf
from scipy.optimize import minimize
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

def cons_1(x):
    return x[0] ** 2 + x[1]

def cons_2(x):
    return x[0] ** 2 - x[1]


def g_1(x):
	return x[0] ** 2 + 6 * x[1] - 36

def g_2(x):
	return x[1] ** 2 - 2 * x[1] - 4 * x[0]

list_of_nonlinear_functions = [g_1, g_2]

class Nonlinear_constraints(object):

    def __init__(self, x_tensor): 
        global list_of_nonlinear_functions
        self._stack = []
        assert (x_tensor.shape.__len__() == 1) , ('shape is not signature: (n, )!')
        self._tensor = x_tensor
        print("tensor: ", x_tensor.__repr__())

        for iter_ in range(len(list_of_nonlinear_functions)):
            self._stack.append(list_of_nonlinear_functions[iter_])

        self._s = len(self._stack)

    def wrapper_conditions_(self):
        def _inner_(vars_x) -> float:
            if(vars_x.shape[0] != self._tensor.shape[0] or vars_x.shape.__len__() != 1):
                assert (False) ('shape is wrong!')

            vars_tf = tf.convert_to_tensor(vars_x)
            stack_return = []
            for iter_ in range(self._s):
                stack_return.append(self._stack[iter_](vars_tf))
            return stack_return # Так или иначе, это стэк list из объектов типа тензоров
        return _inner_


    def jacobian_of_constraints(self):
        global list_of_nonlinear_functions 
        stack_ = []
        with tf.GradientTape(persistent = True) as g:
                g.watch((self._tensor))
                for iter_ in range(self._s):
                    fn_ = list_of_nonlinear_functions[iter_](self._tensor)
                    stack_.append(g.gradient(fn_ , self._tensor))

        return stack_

    def wrapper_jacobian(self):
        def _inner_(vars_x) -> float: 
            if(vars_x.shape[0] != self._tensor.shape[0] or vars_x.shape.__len__() != 1):
                assert (False) ('shape is wrong!')
            vars_tf = tf.convert_to_tensor(vars_x)
            self._tensor = vars_tf 

            return self.jacobian_of_constraints()

        return _inner_

    def hessian_of_constraints(self):
        global list_of_nonlinear_functions
        matrix_total = []
        for iter_ in range(self._s): 
            matrix_current = []
            with tf.GradientTape(persistent = True) as g: 
                g.watch(self._tensor)
                fn_current = list_of_nonlinear_functions[iter_](self._tensor)
                grad_ = g.gradient(fn_current, self._tensor)
                list_current = []
                for jter_ in range(self._tensor.shape[0]):
                    grad_grad_ = g.gradient(grad_[jter_], self._tensor)
                    list_current.append(grad_grad_)

                matrix_current.append(list_current) 

            matrix_total.append(matrix_current)
            matrix_total_tf = tf.convert_to_tensor(matrix_total)

        new_matrix_tf = [] 
        for iter_ in range(self._s):
            new_matrix_tf.append(matrix_total_tf[iter_][0])

        return new_matrix_tf 

    def wrapper_hessians_matrix(self):
        def _inner_(vars_x) -> float:
            if(vars_x.shape[0] != self._tensor.shape[0] or vars_x.shape.__len__() != 1):
                assert (False) ('shape is wrong!')
            vars_tf = tf.convert_to_tensor(vars_x)
            self._tensor = vars_tf

            return self.hessian_of_constraints() 
        return _inner_


def wr_cons_f(nonlinear_constraints):
    def cons_f(x):
        wrapper_obj = nonlinear_constraints.wrapper_conditions_()
        wrapper_obj = list(map(lambda x: x.numpy(), wrapper_obj(x)))
        return wrapper_obj
    return cons_f

def wr_cons_J(nonlinear_constraints):
    def cons_J(x):
        funct_obj = nonlinear_constraints.wrapper_jacobian()
        funct_obj = list(map(lambda x: list(x.numpy()), funct_obj(x)))
        return funct_obj
    return cons_J

def wr_cons_H(nonlinear_constraints):
    def cons_H(x, v): 
        if(nonlinear_constraints._s != v.shape[0] or v.shape.__len__() != 1):
            assert(False) ('shape of v vector is wrong!')

        wrapper_obj = nonlinear_constraints.wrapper_hessians_matrix()
        wrapper_obj = wrapper_obj(x) 
        total_ = 0 
        for iter_ in range(nonlinear_constraints._s):
            total_ += v[iter_] * wrapper_obj[iter_].numpy()

        return total_
    return cons_H


