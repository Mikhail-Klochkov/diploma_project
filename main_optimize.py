
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
from mpl_toolkits.mplot3d import Axes3D
from class_Markovitz import Markovitz_theory, _wrapper_marcovitz, wrapper_fuzzy_constraints_
from scipy.optimize import minimize

#### test simple example ####

########## Markovitz test ##########
model_ = Markovitz_theory() # создаём объект, который содержит наш функционал и параметры модели
model_._initial_model_param_() # Инициализация начальных параметров, делается только один раз

bounds_mark = model_._bounds_() # create a object of bounds condition
W_0 = model_.weights.numpy() # tensor object Variable мы его в numpy переводим(мы его передаём в сам minimize)
### Отличие состоит в выборе нашего нового функционала ###
functionals_Markovitz = wrapper_fuzzy_constraints_(model_)

foo_markovitz_obj = function(W_0[:, 0], functionals_Markovitz) # Так как в классе модели марковица вектор представлен в виде (num, 1) -> (num, ) 
function_mark = foo_markovitz_obj._wrapper()
function_mark_grad = foo_markovitz_obj.wrapper_grad()
function_mark_hess = foo_markovitz_obj.wrapper_hessian()

Q = 0.100 # Минимальная прибль, которая нас интересует 
model_.level_risks = Q
linear_const_mark = model_._linear_constraints_(Q) # create a oblect LinearConstraints
### Вызываем решатель ###
res_Markovitz = minimize(function_mark, W_0[:, 0], method = 'trust-constr', jac = function_mark_grad,
		                        hess = function_mark_hess, constraints = [linear_const_mark],
		                        options = {'verbose': 1}, bounds = bounds_mark)

print('\n answer ' + str(Q) + 'risks: \n ', res_Markovitz.x)
# Необходимо точно обновить веса, чтобы визуализировать наше решение среди большого числа сгенерированных портфелей
model_.weights = tf.Variable(res_Markovitz.x[:, np.newaxis])
model_.plot_a_convex_field(stack_data = None) # Построить без большого количества повторений для одного уровня риска Q
print(model_._volatility_().numpy(), model_.__return_total__().numpy(),
                Q, np.sum(model_.weights.numpy()))
### Мы хотим получить значения как менялись наши значения параметров ### 
evolution_param = foo_markovitz_obj.stack_arg

def _evolution_plot(evol_param):
	x_1 = [a[0] for a in evol_param]
	x_2 = [a[1] for a in evol_param]
	x_3 = [a[2] for a in evol_param]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(x_1, x_2, x_3, label = 'parametric curve')
	ax.set_xlabel("w_1")
	ax.set_ylabel("w_2")
	ax.set_zlabel("w_3")
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_zlim(0, 1)
	ax.view_init(30, 100)
	plt.savefig('evolve.png')

_evolution_plot(evolution_param)