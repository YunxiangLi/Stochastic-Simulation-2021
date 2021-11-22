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
















