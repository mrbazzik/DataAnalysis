
# coding: utf-8

# Подключаем необходимые библиотеки:

# In[49]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import math
get_ipython().magic(u'matplotlib inline')


# Будем моделировать распределение Парето:

# # Распределение Парето

# Сгенерируем выборку объёма 1000 из распределения  Парето (с  $x_m = 1, k = 3$):

# Функция плотности $pdf(x) = 3 / (x^4)$

# In[50]:

pareto_rv = sts.pareto(3, 1)
sample = []
sample = pareto_rv.rvs(1000)
print sample


# Строим гистограмму выборки, поверх гистограммы рисуем теоретическую плотность распределения:

# In[51]:

x = np.linspace(1, 30, 50)
pdf = pareto_rv.pdf(x)
plt.plot(x, pdf, label = 'theoretical pdf', alpha = 2)
plt.legend()
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

# на том же графике построим гистограмму выборки:

plt.hist(sample, normed = True);


# Оценим распределение выборочного среднего нашей случайной величины при разных объёмах выборок:

# Для этого при трёх n (5, 10, 50) сгенерируем 1000 выборок объёма n и построим гистограммы распределений их выборочных средних.
# 

# In[57]:

n = 5
x = np.zeros(1000)
sample = pareto_rv.rvs(1000 * n)
for i in range(1000):
    for j in range(n):
        x[i] += sample[i + j]
    x[i] /= n
#построили массив из выборочных средних

#Строим гистограмму:
plt.hist(x, normed = True)
plt.ylabel('number of samples')
plt.xlabel('$x$');


# In[58]:

n = 10
x = np.zeros(1000)
sample = pareto_rv.rvs(1000 * n)
for i in range(1000):
    for j in range(n):
        x[i] += sample[i + j]
    x[i] /= n
#построили массив из выборочных средних

#Строим гистограмму:
plt.hist(x, normed = True)
plt.ylabel('number of samples')
plt.xlabel('$x$');


# In[59]:

n = 50
x = np.zeros(1000)
sample = pareto_rv.rvs(1000 * n)
for i in range(1000):
    for j in range(n):
        x[i] += sample[i + j]
    x[i] /= n
#построили массив из выборочных средних

#Строим гистограмму:
plt.hist(x, normed = True)
plt.ylabel('number of samples')
plt.xlabel('$x$');


# Используя теоретические данные, вычислим значения математического ожидания, дисперсии, медианы и моды для распределения Парето с $x_m = 1, k = 3$
# 

# In[60]:

x_m = 1.0
k = 3.0
Ex = k * x_m / (k - 1)
med = x_m * math.sqrt(2)
mod = x_m
Dx = ((x_m / (k - 1)) ** 2) * k / (k - 2)
print 'Математическое ожидание: ', Ex
print 'Медиана: ', med
print 'Мода: ', mod
print 'Дисперсия: ', Dx


# Посчитаем значения параметров нормальных распределений, которыми, согласно центральной предельной теореме, приближается распределение выборочных средних:

# In[62]:

n = 5
mu = Ex
sigma = math.sqrt(Dx / n )
print 'n =', n, ':' 'N(',  mu, ',',sigma, ')'
n = 10
mu = Ex
sigma = math.sqrt(Dx / n )
print 'n =', n, ':' 'N(',  mu, ',',sigma, ')'
n = 50
mu = Ex
sigma = math.sqrt(Dx / n )
print 'n =', n, ':' 'N(',  mu, ',',sigma, ')'


# Для всех выборочных средних нормальное распределение будет с параметрами: $\mu = Ex$, $\sigma^2 = Dx / n$

# Поверх каждой гистограммы нарисуем плотность соответствующего нормального распределения

# In[75]:

n = 5
x = np.zeros(1000)
sample = pareto_rv.rvs(1000 * n)
for i in range(1000):
    for j in range(n):
        x[i] += sample[i + j]
    x[i] /= n
#построили массив из выборочных средних

#Строим гистограмму:
plt.hist(x, normed = True)

#нормальное распределние:
mu = Ex + 1
sigma = math.sqrt(Dx / n)
norm_rv = sts.norm(mu, sigma)

#Строим график плотности распределения:
x = np.linspace(0, 8, 100)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf, label = 'theoretical pdf', alpha = 2)
plt.legend()
plt.ylabel('$f(x)$')
plt.xlabel('$x$');


# In[76]:

n = 10
x = np.zeros(1000)
sample = pareto_rv.rvs(1000 * n)
for i in range(1000):
    for j in range(n):
        x[i] += sample[i + j]
    x[i] /= n
#построили массив из выборочных средних

#Строим гистограмму:
plt.hist(x, normed = True)

#нормальное распределние:
mu = Ex + 1
sigma = math.sqrt(Dx / n)
norm_rv = sts.norm(mu, sigma)

#Строим график плотности распределения:
x = np.linspace(0, 8, 100)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf, label = 'theoretical pdf', alpha = 2)
plt.legend()
plt.ylabel('$f(x)$')
plt.xlabel('$x$');


# In[78]:

n = 50
x = np.zeros(1000)
sample = pareto_rv.rvs(1000 * n)
for i in range(1000):
    for j in range(n):
        x[i] += sample[i + j]
    x[i] /= n
#построили массив из выборочных средних

#Строим гистограмму:
plt.hist(x, normed = True)

#нормальное распределние:
mu = Ex + 1
sigma = math.sqrt(Dx / n)
norm_rv = sts.norm(mu, sigma)

#Строим график плотности распределения:
x = np.linspace(0, 7, 100)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf, label = 'theoretical pdf', alpha = 2)
plt.legend()
plt.ylabel('$f(x)$')
plt.xlabel('$x$');


# Как мы видим, выборочные средние достаточно близки к нормальному распределению с соответствующими коэффициентами. Точно аппроксимации увеличивается с ростом n 

# In[ ]:



