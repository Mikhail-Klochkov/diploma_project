import tensorflow.compat.v2 as tf
from scipy.optimize import minimize
import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

@tf.function
def input_tens(tens):
    tf.print(tens)

def rosen(x): # numpy example of function
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)


@tf.function
def input_(x_tens):
    tf.print(x_tens)

def rosen_tf(x_tens):
    return tf.reduce_sum(100.0 * (x_tens[1:] - x_tens[:-1] ** 2) ** 2 + (1 - x_tens[:-1]) ** 2)


## define a nonlinear condition functions where is we translate a tensortype

def cons_1(x):
    return x[0] ** 2 + x[1]

def cons_2(x):
    return x[0] ** 2 - x[1]

list_of_nonlinear_functions = [cons_1, cons_2] #

## ________end_________

class Nonlinear_constraints(object):

    def __init__(self, x_tensor): # we transfer a x_tensor
        global list_of_nonlinear_functions
        self._stack = []
        ## Here we should write a all nonlinear functions:
        assert (x_tensor.shape.__len__() == 1) , ('shape is not signature: (n, )!')


        self._tensor = x_tensor
        print("tensor: ", x_tensor.__repr__())
        # we work with 2 dims
        for iter_ in range(len(list_of_nonlinear_functions)):
            self._stack.append(list_of_nonlinear_functions[iter_])

        self._s = len(self._stack)

    def wrapper_conditions_(self):
        def _inner_(vars_x) -> float:
            ## Вообще нет смысла обновлять значения
            if(vars_x.shape[0] != self._tensor.shape[0] or vars_x.shape.__len__() != 1):
                assert (False) ('shape is wrong!')

            vars_tf = tf.convert_to_tensor(vars_x)
            stack_return = []
            for iter_ in range(self._s):
                stack_return.append(self._stack[iter_](vars_tf))
            return stack_return # Так или иначе, это стэк list из объектов типа тензоров
        return _inner_


    def jacobian_of_constraints(self):
        global list_of_nonlinear_functions # Тут у нас лежат наши функции нелинейные
        stack_ = []
        with tf.GradientTape(persistent = True) as g:
                g.watch((self._tensor))
                for iter_ in range(self._s):
                    fn_ = list_of_nonlinear_functions[iter_](self._tensor)
                    stack_.append(g.gradient(fn_ , self._tensor))

        return stack_

    def wrapper_jacobian(self):
        def _inner_(vars_x) -> float: # vars_s should be a np.array object
            if(vars_x.shape[0] != self._tensor.shape[0] or vars_x.shape.__len__() != 1):
                assert (False) ('shape is wrong!')
            vars_tf = tf.convert_to_tensor(vars_x)
            self._tensor = vars_tf # we should to change, because  we call jacobian method with automaticly computetion jacobian with self._tensor

            return self.jacobian_of_constraints()

        return _inner_

    def hessian_of_constraints(self):
        global list_of_nonlinear_functions
        matrix_total = []
        for iter_ in range(self._s): # по всем таким функциям делаем проход для рассчёта Hessiana
            matrix_current = []
            with tf.GradientTape(persistent = True) as g: # В рамках одной такой
                g.watch(self._tensor)
                fn_current = list_of_nonlinear_functions[iter_](self._tensor)
                grad_ = g.gradient(fn_current, self._tensor)
                list_current = []
                for jter_ in range(self._tensor.shape[0]):
                    grad_grad_ = g.gradient(grad_[jter_], self._tensor)
                    list_current.append(grad_grad_)

                matrix_current.append(list_current) # На выходе должна быть матрица n*n от размерности

            matrix_total.append(matrix_current)
            matrix_total_tf = tf.convert_to_tensor(matrix_total)

        new_matrix_tf = [] # Чтобы убрать одну размерность!
        for iter_ in range(self._s):
            new_matrix_tf.append(matrix_total_tf[iter_][0])

        return new_matrix_tf # Он возвращает лист с перечисленными тензорами, которые есть Гессианы для каждой нелинейной функции ограничения

    def wrapper_hessians_matrix(self):
        def _inner_(vars_x) -> float:
            if(vars_x.shape[0] != self._tensor.shape[0] or vars_x.shape.__len__() != 1):
                assert (False) ('shape is wrong!')
            vars_tf = tf.convert_to_tensor(vars_x)
            self._tensor = vars_tf

            return self.hessian_of_constraints() # list of tensors hessians
        return _inner_


class function(tf.Module):

    # аттрибут для вывода значений вектора x - как он у нас меняется
    # в процессе иттераций алгоритма scipy

    def compute_gradient_optional(self, functional_obj):
            with tf.GradientTape() as g:
                g.watch(self._tensor)
                res = functional_obj(self._tensor)
                grad_ = g.gradient(res, self._tensor)

            return grad_

    def __init__(self, x_numpy):
        if(x_numpy.shape.__len__() != 1):
            assert (1 != 1) ('shape is not signature : (n, )!')
        else:
            self._tensor = tf.Variable(x_numpy)
            self._times_call_change = 0

    def __call__(self, some_parameter = 0): # __call__ function and convert in tensor

        return tf.convert_to_tensor(rosen(self._tensor.numpy()))

    def __repr__(self):
        return (self._tensor.__repr__(), self._another.__repr__())

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
            print('current value of x: ', self._tensor.numpy())

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
                global rosen_tf
                with tf.GradientTape() as g:
                    g.watch((self._tensor))
                    f_a = rosen_tf(self._tensor)
                    #f_a = tf.reduce_sum(100.0 * (self._tensor[1:] - self._tensor[:-1] ** 2) ** 2 + (1 - self._tensor[:-1]) ** 2)
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
        global rosen_tf
        with tf.GradientTape(persistent = True) as g:
            g.watch(self._tensor)
            f_a = rosen_tf(self._tensor)
            #f_a = tf.reduce_sum(100.0 * (self._tensor[1:] - self._tensor[:-1] ** 2) ** 2 + (1 - self._tensor[:-1]) ** 2)
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


x_tens_ = tf.constant(np.full((2,), 2.))
foo_ = function(x_tens_)
function_ = foo_._wrapper()
funct_grad = foo_.wrapper_grad()
funct_hessian = foo_.wrapper_hessian()

from scipy.optimize import minimize
x_0 = np.array([1.3 , 0.7])

#res_ = minimize(function_, x_0, method = 'nelder-mead', options = {'xtol': 1e-8, 'disp': True})
#print(res_.x)
#res_an_ = minimize(function_, x_0, method = 'BFGS', jac = funct_grad, options = {'disp': False})

#print(res_an_)

res_another = minimize(function_, x_0, method = 'Newton-CG', jac = funct_grad,hess = funct_hessian,options = {'xtol' : 1e-8, 'disp': True})
print(res_another)


x_ = tf.Variable(np.full((2, ), 2.), dtype = tf.float64)
nonlinear_constraints = Nonlinear_constraints(x_) # create a object Nonlinear

def cons_f(x):
    global nonlinear_constraints
    wrapper_obj = nonlinear_constraints.wrapper_conditions_()
    wrapper_obj = list(map(lambda x: x.numpy(), wrapper_obj(x)))
    return wrapper_obj

nonlinear_constraints.jacobian_of_constraints()

def cons_J(x):
    global nonlinear_constraints
    funct_obj = nonlinear_constraints.wrapper_jacobian()
    funct_obj = list(map(lambda x: list(x.numpy()), funct_obj(x)))
    return funct_obj

cons_J(x_.numpy()) # returned the list object with np.float type
Hessian_ = nonlinear_constraints.hessian_of_constraints()

def cons_H(x, v): # x - np.array() v - np.array() also
    global nonlinear_constraints
    if(nonlinear_constraints._s != v.shape[0] or v.shape.__len__() != 1):
        assert(False) ('shape of v vector is wrong!')

    wrapper_obj = nonlinear_constraints.wrapper_hessians_matrix()
    wrapper_obj = wrapper_obj(x) # _inner_(vars_x) -> float: по факту тут вычисленный list с tensors hessians
    total_ = 0 # H(x, v) как в статье
    for iter_ in range(nonlinear_constraints._s):
        total_ += v[iter_] * wrapper_obj[iter_].numpy()

    return total_

cons_H(x_.numpy(), np.full((2, ), 3.))
bounds = Bounds([0, -0.5], [1.0, 2.0])
linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1],
                                     [1,1])
nonlinear_constraints_ = NonlinearConstraint(cons_f, -np.inf, 1,
                                             jac = cons_J, hess = cons_H)

res_constraints_ = minimize(function_, x_0, method = 'trust-constr', jac = funct_grad,
               hess = funct_hessian,
               constraints = [linear_constraint, nonlinear_constraints_],
               options = {'verbose': 1}, bounds = bounds)
print(res_constraints_.x)

