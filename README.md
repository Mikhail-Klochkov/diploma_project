# diploma_project
Данные наброски очень грязные и были сделаны исключительно с целью собственного пользования.
Актуальность данных файлов уже сомнительная.

Реализован некоторый удобный для авторского пользования инструмент (class_nonlinear_constraints.py, class_function.py) для поиска множества оптимума в смешанных задачах оптимизации, которые требуют расчёта первых, а также вторых производных (для методов второго порядка). Реализован функционал по рассчёту 1-2 производных (Яобиан, Гессиан) методами tensorflow. Чтобы автору только приходилось задавать задачу условной оптимизации, в виде набора целевых функций, а далее расчёт (Якобианаб Гессиана) происходил автоматически!!! Далее использовался интструментарий scipy.optimize, куда далее уже подавались все необходимые (рассчитанные 1-2 производные).

Остальные файлы содержат более содержательную часть проблемы дипломной работы. 

