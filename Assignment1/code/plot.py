# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:19:40 2021

@author: Yunxiang
"""

import numpy as np
import csv
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# plot
path = './data/'
data_random = pd.read_csv(path + 'test_s_random.csv', header=None, names = ['iter', 'nsamples', 'run', 'time', 'area'])
data_random['method'] = 'Random'
data_lhs = pd.read_csv(path + 'test_s_lhs.csv', header=None, names = ['iter', 'nsamples', 'run', 'time', 'area'])
data_lhs['method'] = 'LHS'
data_ortho = pd.read_csv(path + 'test_s_ortho.csv', header=None, names = ['iter', 'nsamples', 'run', 'time', 'area'])
data_ortho['method'] = 'Orthogonal'

data = pd.concat([data_random, data_lhs, data_ortho])


data_anti = pd.read_csv(path + 'test_s_anti_random.csv', header=None, names = ['iter', 'nsamples', 'run', 'time', 'area'])
data_anti['method'] = 'Random(Antithetic)'
# data = pd.concat([data_random, data_anti])




# investigate i
sns.lineplot(x = "iter", y = "area", data=data_random)
A_i = np.mean(data_random[data_random['iter'] == 2000]['area'])
A = np.mean(data_random[data_random['iter'] == 10000]['area'])
A_std = np.std(data_random[data_random['iter'] == 10000]['area'])
plt.axhline(y = A, color='r', linestyle='--', label=f'{A:.4f}')
plt.legend(loc = 'right')
plt.xlabel('Number of iterations')
plt.ylabel('Area of Mandelbrot Set')
plt.savefig('investigate_i', dpi = 1000)


# investigate i further precise
sns.lineplot(x = "iter", y = "area", data=data_random)
plt.axhline(y = 1.5068, color='r', linestyle='--', label='1.5068')
plt.legend(loc = 'right')
plt.xlabel('Number of iterations')
plt.ylabel('Area of Mandelbrot Set')
plt.savefig('investigate_i_precise.png', dpi = 1000)


# investigate i error
A = np.mean(data_random[data_random['iter'] == 10000]['area'])
iters = np.arange(100,10001,100)
def df_error(df):
    means = []
    for i in iters:
        mean = np.mean(df[df['iter'] == i]['area'])
        means.append(mean)
    errors = []
    for mean in means:
        error = abs(mean - A)
        errors.append(error)
    return errors
plt.plot(iters, df_error(data_random), 'r')
plt.xlabel('Number of iterations')
plt.ylabel('Absolute error')
plt.savefig('investigate_i_error', dpi = 1000)


# investigate i time
sns.lineplot(x = "iter", y = "time", color = 'g',data=data_random)
plt.xlabel('Number of iterations')
plt.ylabel('Time of each simulation')
plt.savefig('investigate_i_time', dpi = 1000)

# investigate s random
sns.lineplot(x = "nsamples", y = "area", data=data_random)
A = np.mean(data_random[data_random['nsamples'] == 10000]['area'])
plt.axhline(y = A, color='r', linestyle='--', label=f'{A:.4f}')
plt.legend(loc = 'upper right')
plt.xlabel('Number of samples')
plt.ylabel('Area of Mandelbrot Set')
plt.savefig('investigate_s', dpi = 1000)


# compare three methods
fig, axes = plt.subplots(nrows=3, ncols=1,sharex = True, figsize = (10,15))

sns.lineplot(ax=axes[0], x = "nsamples", y = "area", data = data_random, color = 'xkcd:blue')
axes[0].set_ylim([1.45, 1.60])
A = np.mean(data_random[data_random['nsamples'] == 10000]['area'])
axes[0].axhline(y = A, color='r', linestyle='--', label=f'{A:.4f}')
axes[0].legend(fontsize = 25)
axes[0].set_title('Pure random sampling', fontsize = 20, color = 'xkcd:blue')
axes[0].set_ylabel('Area of Mandelbrot Set', fontsize = 20)

sns.lineplot(ax=axes[1], x = "nsamples", y = "area", data = data_lhs, color = 'xkcd:orange')
axes[1].set_ylim([1.45, 1.60])
A = np.mean(data_lhs[data_lhs['nsamples'] == 10000]['area'])
axes[1].axhline(y = A, color='r', linestyle='--', label=f'{A:.4f}')
axes[1].legend(fontsize = 25)
axes[1].set_title('Latin hypercube sampling', fontsize = 20, color = 'xkcd:orange')
axes[1].set_ylabel('Area of Mandelbrot Set', fontsize = 20)

sns.lineplot(ax=axes[2], x = "nsamples", y = "area", data = data_ortho, color = 'xkcd:green')
axes[2].set_ylim([1.45, 1.60])
A = np.mean(data_ortho[data_ortho['nsamples'] == 10000]['area'])
axes[2].axhline(y = A, color='r', linestyle='--', label=f'{A:.4f}')
axes[2].legend(fontsize = 25)
axes[2].set_title('Orthogonal sampling', fontsize = 20, color = 'xkcd:green')
axes[2].set_ylabel('Area of Mandelbrot Set', fontsize = 20)
axes[2].set_xlabel('Number of samples', fontsize = 20)

fig.tight_layout()
fig.savefig('sampling_methods', dpi = 1000)



# std, error, time three methods
fig, axes = plt.subplots(nrows=3, ncols=1, sharex = True, figsize = (10,15))

nsamples = np.arange(100,10001,100)
def df_std(df):
    stds = []
    for n in nsamples:
        std = np.std(df[df['nsamples'] == n]['area'])
        stds.append(std)
    return stds

std_random = df_std(data_random)
std_lhs = df_std(data_lhs)
std_ortho = df_std(data_ortho)

axes[0].plot(nsamples, std_random, 'xkcd:blue', label = 'Random')
axes[0].plot(nsamples, std_lhs, 'xkcd:orange', label = 'LHS')
axes[0].plot(nsamples, std_ortho, 'xkcd:green', label = 'Orthogonal')
axes[0].legend(fontsize = 20)
axes[0].set_ylabel('Standard Deviation', fontsize = 20)

nsamples = np.arange(100,10001,100)
def df_error(df):
    # A = np.mean(df[df['nsamples'] == 10000]['area'])
    A = 1.506484
    print(A)
    means = []
    for n in nsamples:
        mean = np.mean(df[df['nsamples'] == n]['area'])
        means.append(mean)
    errors = []
    for mean in means:
        error = abs(mean - A)
        errors.append(error)
    return errors

axes[1].plot(nsamples, df_error(data_random), 'xkcd:blue',label = 'Random')
axes[1].plot(nsamples, df_error(data_lhs), 'xkcd:orange',label = 'LHS')
axes[1].plot(nsamples, df_error(data_ortho), 'xkcd:green',label = 'Orthogonal')
axes[1].set_ylabel('Absolute error',fontsize = 20)
axes[1].legend(fontsize = 20)

sns.lineplot(ax = axes[2], x = "nsamples", y = "time", data = data, hue = 'method')
axes[2].set_xlabel('Number of samples',fontsize = 20)
axes[2].set_ylabel('Time per simulation',fontsize = 20)
axes[2].legend(fontsize = 20)
fig.tight_layout()
fig.savefig('std_error_time', dpi = 1000)



# table stats
r = data_random[data_random['nsamples'] == 10000]
l = data_lhs[data_lhs['nsamples'] == 10000]
o =  data_ortho[data_ortho['nsamples'] == 10000]
a = data_anti[data_ortho['nsamples'] == 10000]
def stats(s):
    area  = s['area']
    time = s['time']
    mean = np.mean(area)
    sd= np.std(area)
    print(f'mean: {mean:.4f}')
    print(f'standard deviation: {sd:.4f}')
    print(f'standard deviation: ({(mean - 1.96*sd/10):.4f}, {(mean + 1.96*sd/10):.4f})')
    print(f'{np.mean(time):.4f}')

# area std, error   antithetic
fig, axes = plt.subplots(nrows=3, ncols=1, sharex = True, figsize = (10,15))

nsamples = np.arange(100,10001,100)

sns.lineplot(ax = axes[0], x = "nsamples", y = "area", hue = 'method', ci =None, data = data)
axes[0].legend(fontsize = 20)
axes[0].set_ylabel('Area', fontsize = 20)

def df_std(df):
    stds = []
    for n in nsamples:
        std = np.std(df[df['nsamples'] == n]['area'])
        stds.append(std)
    return stds

axes[1].plot(nsamples, df_std(data_random), 'xkcd:blue', label = 'Random')
axes[1].plot(nsamples, df_std(data_anti), 'xkcd:orange', label = 'Random(Antithetic)')
axes[1].legend(fontsize = 20)
# axes[1].set_yscale('log')
axes[1].set_ylabel('Standard Deviation', fontsize = 20)

nsamples = np.arange(100,10001,100)
def df_error(df):
    # A = np.mean(df[df['nsamples'] == 10000]['area'])
    A = 1.506484
    print(A)
    means = []
    for n in nsamples:
        mean = np.mean(df[df['nsamples'] == n]['area'])
        means.append(mean)
    errors = []
    for mean in means:
        error = abs(mean - A)
        errors.append(error)
    return errors

axes[2].plot(nsamples, df_error(data_random), 'xkcd:blue',label = 'Random')
axes[2].plot(nsamples, df_error(data_anti), 'xkcd:orange',label = 'Random(Antithetic)')
axes[2].set_ylabel('Absolute error',fontsize = 20)
axes[2].legend(fontsize = 20)
axes[2].set_xlabel('Number of samples',fontsize = 20)

fig.savefig('anti', dpi = 1000)
















