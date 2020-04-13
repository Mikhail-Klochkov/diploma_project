

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

Q = 0.125 # Минимальная прибль, которая нас интересует 
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