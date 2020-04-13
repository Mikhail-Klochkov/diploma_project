
### Вставить после комментария test simple example ###
def funct_another(x):
	return -((x[0] - 1.) ** 2 + x[1] ** 2)

def rosen_tf(x_tens):
    return tf.reduce_sum(100.0 * (x_tens[1:] - x_tens[:-1] ** 2) ** 2 + (1 - x_tens[:-1]) ** 2)

number_of_task = 2 # Параметр для запуска одного из простых тестов

######## test = 1 ########

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
########## test = 2 ##########
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