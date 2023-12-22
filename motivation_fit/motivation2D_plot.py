import numpy as np
import matplotlib.pyplot as plt

import math

def square_root_as_int(x):
  return int(math.sqrt(x))

def factors_of_(number):
  factors = []
  for x in range(square_root_as_int(number) + 1,0,-1):
    if number % x == 0:
      return x, number/x

def boundaries(x1,x2):
    return 2*x1*x2+x1+x2

from scipy.optimize import curve_fit

a = np.load('random_step2D_example.npy')
mean = np.mean(a, axis=0)
std = np.std(a, axis=0)

x_origin = range(10, 260, 10)

bounds = []
x_final = []
for i in range(len(x_origin)):
    x1, x2 = factors_of_(x_origin[i])

    b = boundaries(x1, x2)
    if b not in bounds:
        bounds.append(b)
        x_final.append(x_origin[i])
    else:
        print(i, x_origin[i])


bounds = np.array(bounds,dtype=np.int_)
order = np.argsort(bounds)

bounds = bounds[order]
a = a[:,order]
x = bounds


def func(x, a, b, c):
    return a * np.exp(b * x) + c


plt.rc('font', size=16)

z1 = np.polyfit(x, mean, 3)
p1 = np.poly1d(z1)
y_pre = p1(x)

popt, pcov = curve_fit(func, x, y_pre, bounds=([1,0.001,-100], [60, 0.1, 10]))

y_pred = [func(i, popt[0], popt[1],popt[2]) for i in x]

p = np.exp(popt[1])
print('p = ', p)

fontsize = 16
plt.plot(x, mean, label='Convergence Step Mean ')
plt.fill_between(x, mean + std, mean - std,
                 facecolor='green',
                 edgecolor='red',
                 alpha=0.3,
                 label='Convergence Step Std')

plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
plt.xticks(size=fontsize)
plt.yticks(size=fontsize)


plt.plot(x, y_pred, 'r--', label='Exponential Curve')
plt.xlabel('number of boundaries', fontsize=fontsize)
plt.ylabel('steps',fontsize=fontsize)
plt.legend(loc=2,fontsize=fontsize)


plt.savefig('motivation2D.png',bbox_inches='tight')
plt.show()

