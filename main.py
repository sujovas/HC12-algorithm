from hc12 import HC12
import objfunc
import numpy as np
import matplotlib.pyplot as plot
import hc12

# dimenze
n_param = 100
# pocet bitu na parametr
n_bit_param = 5
# rozsah hodnot - def.obor
dod_param=[-5.12, 5.12]
# dod_param=[-5, 10]
# dod_param=[-500, 500]
#times = opakovani
times = 1
# #objective fcn
func = objfunc.rastrigin
# func = objfunc.rosenbrock
# func = objfunc.schwefel

hc12_instance = HC12(n_param, n_bit_param, dod_param)
# print('rows', hc12_instance.rows)
x, fx = hc12_instance.run(func, times, )
# index nejlepsiho behu
best_idx = np.argmin(fx)
iteration = []
# print(f'Minimal value: {min(hc12_instance.P)}')
for i in range(len(hc12_instance.P)):
    iteration.append(i)
plot.step(iteration, hc12_instance.P)
plot.xlabel('Iterations')
plot.xlim(0, len(hc12_instance.P) - 1)
plot.ylim(0)
plot.grid(True)
plot.title('Best solution for %dD' % (n_param))
plot.show()

print(f'best: x={x[best_idx, :]}, fx ={fx[best_idx]}')
# return np.min(hc12_instance.T), np.min(hc12_instance.P)
