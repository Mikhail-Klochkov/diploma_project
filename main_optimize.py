
import tensorflow.compat.v2 as tf
from scipy.optimize import minimize
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from class_function import function
from class_nonlinear_constraints import Nonlinear_constraints, \
											wr_cons_f, wr_cons_J, wr_cons_H, \
											g_1, g_2, cons_1, cons_2, \
											list_of_nonlinear_functions

from class_Markovitz import Markovitz_theory, _wrapper_marcovitz
from scipy.optimize import minimize

def funct_another(x):
	return -((x[0] - 1.) ** 2 + x[1] ** 2)

def rosen_tf(x_tens):
    return tf.reduce_sum(100.0 * (x_tens[1:] - x_tens[:-1] ** 2) ** 2 + (1 - x_tens[:-1]) ** 2)

number_of_task = 2

if number_of_task == 1:
	x_0 = np.array([1.3 , 0.7])
	x_tens_ = tf.Variable(np.array([1.3, 0.7]))
	foo_ = function(x_tens_, rosen_tf)
	function_ = foo_._wrapper()
	funct_grad = foo_.wrapper_grad()
	funct_hessian = foo_.wrapper_hessian()

	bounds = Bounds([0, -0.5], [1.0, 2.0])
	linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1],
	                                     [1,1])

	x_ = tf.Variable(np.full((2, ), 2.), dtype = tf.float64)
	nonlinear_constraints = Nonlinear_constraints(x_)

	nonlinear_constraints_ = NonlinearConstraint(wr_cons_f(nonlinear_constraints), -np.inf, 1,
	                                             jac = wr_cons_J(nonlinear_constraints), hess = wr_cons_H(nonlinear_constraints))

	res_constraints_ = minimize(function_, x_0, method = 'trust-constr', jac = funct_grad,
	               hess = funct_hessian,
	               constraints = [linear_constraint],
	               options = {'verbose': 1}, bounds = bounds)
	print(res_constraints_.x)
##########
if number_of_task == 2:

	x_tens_2 = tf.Variable(np.array([0.0,2.]))
	x_0_2 = x_tens_2.numpy()
	foo_2 = function(x_tens_2, funct_another)
	function_2 = foo_2._wrapper()
	funct_grad_2 = foo_2.wrapper_grad()
	funct_hessian_2 = foo_2.wrapper_hessian()

	bounds_2 = Bounds([0.0, 0.0], [np.inf, np.inf])
	nonlinear_constraints2 = Nonlinear_constraints(x_tens_2)
	nonlinear_constraints_2 = NonlinearConstraint(wr_cons_f(nonlinear_constraints2), -np.inf , 0.0 ,
	                                             jac = wr_cons_J(nonlinear_constraints2), hess = wr_cons_H(nonlinear_constraints2))

	res_constraints_2 = minimize(function_2, x_0_2, method = 'trust-constr', jac = funct_grad_2,
	               hess = funct_hessian_2,
	               constraints = [nonlinear_constraints_2],
	               options = {'verbose': 1}, bounds = bounds_2)
	print("\n Result \n " , res_constraints_2.x, '\n')
##########

model_ = Markovitz_theory() # создаём объект, который содержит наш функционал и параметры модели
model_._initial_model_param_() # Инициализация начальных параметров, делается только один раз

bounds_mark = model_._bounds_() # create a object of bounds condition
W_0 = model_.weights.numpy() # tensor object Variable мы его в numpy переводим(мы его передаём в сам minimize)

functionals_Markovitz = _wrapper_marcovitz(model_) # return a functional_obj
### Повторяем как было выше для обычной минимизации ###
foo_markovitz_obj = function(W_0[:, 0], functionals_Markovitz) # Так как в классе модели марковица вектор представлен в виде (num, 1) -> (num, ) 
function_mark = foo_markovitz_obj._wrapper()
function_mark_grad = foo_markovitz_obj.wrapper_grad()
function_mark_hess = foo_markovitz_obj.wrapper_hessian()
### Конец создания необходимых функциональных объектов ###

### созданеи объектов типа linear_const и bounds ###
def stack_another_risk_results(model_):
	# стоит сбросить найденное новое значение весов предыдущей модели на риски
	range_ = np.linspace(0.01, 0.175, 10)
	stack_data = []
	for item in range_:
		weights_generate = np.random.random(model_.weights.shape[0])
		model_.weights = tf.Variable((weights_generate / np.sum(weights_generate))[:, np.newaxis],
												dtype = tf.float64, name = 'weights')

		Q = item # Риск, допустимый, который нас устраивает 
		model_.level_risks = Q
		linear_const_mark = model_._linear_constraints_(Q) # create a oblect LinearConstraints
		### Вызываем решатель ###
		res_Markovitz = minimize(function_mark, W_0[:, 0], method = 'trust-constr', jac = function_mark_grad,
		                        hess = function_mark_hess, constraints = [linear_const_mark],
		                        options = {'verbose': 1}, bounds = bounds_mark)
		print('\n answer ' + str(Q) + 'risks: \n ', res_Markovitz.x)
		### change our new parameter ###
		model_.weights = tf.Variable(res_Markovitz.x[:, np.newaxis])
		model_.weights

		print("after minimization of Markovitz theory: \n" , model_.__return_total__(), model_._volatility_(), '\n')
		stack_data.append([Q, model_._volatility_().numpy(), model_.__return_total__().numpy(), model_.weights.numpy()])

	return stack_data
stack_data = stack_another_risk_results(model_)
model_.plot_a_convex_field(stack_data = stack_data)
