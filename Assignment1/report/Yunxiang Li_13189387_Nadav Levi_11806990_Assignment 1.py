#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# adapted from http://numba.pydata.org/numba-doc/0.35.0/user/examples.html
# mandelbrot fractal figure 

@jit
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return 255

@jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    width = image.shape[0]
    height = image.shape[1]


    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[x, y] = color

    return

image = np.zeros((6000 * 2,  5000* 2), dtype=np.uint8)
s = timer()
create_fractal(-2.0, 1.0, -1.25, 1.25, image, 100)
e = timer()
print(e - s)
plt.axis('off')
plt.imshow(image,cmap='gist_earth')
plt.savefig('mandelbrot.png', dpi = 1000, bbox_inches='tight', pad_inches=0)


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:40:42 2021

@author: Yunxiang
"""
import numpy as np
import random
import csv
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

# Investigating Area 

@njit
def in_mandelbrot(c, iters):
    """
    Whether a given complex number c is in the Mandelbrot Set. 
    """
    z = 0
    for i in range(iters):
        # If it's in the set
        if abs(z) > 2:
            return False
        # update z
        z = z * z + c
    return True

def area_mandelbrot(rmin, rmax, imin, imax, iters, nsamples, samples):
    """ 
    The ratio between the number of samples in the Mandelbrot set and 
    all samples taken multiplied by the total search area
    yields an estimation of the area of Mandelbrot set.
    """
    in_pts = 0
    total_pts = nsamples
    total_area = (rmax - rmin) * (imax - imin)
    for c in samples:
        if in_mandelbrot(c, iters):
            in_pts += 1
    area = in_pts / total_pts * total_area
    
    return area

def random_sample(rmin, rmax, imin, imax, nsamples):
    """
    Generates a list of pure random complex numbers.
    """
    samples = []
    for n in range(nsamples):
        c = complex(random.uniform(rmin, rmax), random.uniform(imin, imax))
        samples.append(c)
    return samples

def lhs_sample(rmin, rmax, imin, imax, nsamples):
    """ 
    Generate a list of random complex numbers through the latin hypercube method.
    1) Define nsample numbers of intervals (n+1 bounds) for real and imaginary axis.
    2) Draw tuples of real and imaginary coordinates in each of their interval.
    3) Randomly shuffle the combinations by imaginary axis.
    """
    real_intervals = np.linspace(rmin, rmax, nsamples + 1)
    im_intervals = np.linspace(imin, imax, nsamples + 1)
    
    pts = np.empty(shape=(nsamples, 2))
    for i in range(nsamples):
        pts[i, 0] = np.random.uniform(real_intervals[i], real_intervals[i + 1])
        pts[i, 1] = np.random.uniform(im_intervals[i], im_intervals[i + 1])

    np.random.shuffle(pts[:, 1])
    samples = [complex(pts[n, 0], pts[n , 1]) for n in range(len(pts))]
    
    return samples

def orthogonal_sample(rmin, rmax, imin, imax, nsamples):
    """
    Generate a list of orthogonally sampled complex numbers.
    """
    subspace = int(np.sqrt(nsamples)) # The grid width of subspace
    lst_r = []
    lst_i = []
    unit_r = (rmax - rmin) / nsamples
    unit_i = (imax - imin) / nsamples
    
    grids_r = np.arange(0, subspace*subspace, dtype=int).reshape((subspace, subspace))
    grids_i = np.arange(0, subspace*subspace, dtype=int).reshape((subspace, subspace))
    np.random.shuffle(grids_r)
    np.random.shuffle(grids_i)
    
    for i in range(subspace):
        for j in range(subspace):
            lst_r.append(rmin +  (grids_r[i][j] + np.random.random()) * unit_r)
            lst_i.append(imin +  (grids_i[j][i] + np.random.random()) * unit_i)
            # print(grids_r[i][j], grids_i[j][i])
    
    samples = [complex(lst_r[i], lst_i[i]) for i in range(len(lst_i))]
    
    return samples

def main():
    """
    Main function for fixed numbers of iterations and samples. 
    """
    rmin, rmax = -2, 0.5
    imin, imax = -1.2, 1.2
    
    iters = 100
    nsamples = 10000
    runs = 100
    
    for run in range(runs):
        t0 = time.time()
        # Variate the sampling method here
        pts = random_sample(rmin, rmax, imin, imax, nsamples)
        area = area_mandelbrot(rmin, rmax, imin, imax, iters, nsamples)
        t = time.time() - t0
        
        # Manage outputfile
        outfilename = "./data/random.csv"
        with open(outfilename, 'a', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([run, t, iters, nsamples, area])
    return 


def investigate_i_s():
    """
    Investigate the optimal number of iterations with fixed number of samples or vice versa.
    """
    rmin, rmax = -2, 0.5
    imin, imax = -1.2, 1.2
    
    iters = [2000]
    # iters = np.arange(100,10001,100)
    # iters = np.arange(3010,4001,10)
    nsamples = [10000]
    # nsamples = np.arange(100,10001,100)
    runs = 900
    
    for i in iters:
        for s in nsamples:
            for run in range(runs):
                t0 = time.time()
                
                # Variate the sampling method here
                pts = orthogonal_sample(rmin, rmax, imin, imax, s)
                
                area = area_mandelbrot(rmin, rmax, imin, imax, i, s, pts)
                t = time.time() - t0
                
                # Manage outputfile
                outfilename = "./data/test_s_ortho.csv"
                
                with open(outfilename, 'a', newline='') as outfile:
                    writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([i, s, run, t, area])
            print(f'iters = {i}, nsmaples = {s} done.')


def Antithetic_random():
    """
    Generates a list of pure random complex numbers.
    """
    rmin, rmax = -2, 0.5
    imin, imax = -1.2, 1.2
    
    iters = [2000]
    # iters = np.arange(100,10001,100)
    # nsamples = [10000]
    nsamples = np.arange(100,10001,100)
    runs = 100
    
    for i in iters:
        for s in nsamples:
            for run in range(runs):
                t0 = time.time()
                pts1 = random_sample(rmin, rmax, imin, imax, s)
                area1 = area_mandelbrot(rmin, rmax, imin, imax, i, s, pts1)
                pts2 = [complex(-1-c.real, -c.imag) for c in pts1]
                area2 = area_mandelbrot(rmin, rmax, imin, imax, i, s, pts2)
                area = (area1 + area2)/2
                t = time.time() - t0
                
                # Manage outputfile
                outfilename = "./data/test_s_anti_random.csv"
                
                with open(outfilename, 'a', newline='') as outfile:
                    writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([i, s, run, t, area])
                    
            print(f'iters = {i}, nsmaples = {s} done.')




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

# data = pd.concat([data_random, data_lhs, data_ortho])


data_anti = pd.read_csv(path + 'test_s_anti_random.csv', header=None, names = ['iter', 'nsamples', 'run', 'time', 'area'])
data_anti['method'] = 'Random(Antithetic)'
data = pd.concat([data_random, data_anti])



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
def stats(s):
    area  = s['area']
    time = s['time']
    mean = np.mean(area)
    sd= np.std(area)
    print(f'mean: {mean:.4f}')
    print(f'standard deviation: {sd:.4f}')
    print(f'standard deviation: ({(mean - 1.96*sd/10):.4f}, {(mean + 1.96*sd/10):.4f})')
    print(f'{np.mean(time):.4f}')
    
    
    
#area std, error   antithetic
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











