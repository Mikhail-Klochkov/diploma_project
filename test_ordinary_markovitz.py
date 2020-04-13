
### Просто вставить после коментария test Markovitz theory ###

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