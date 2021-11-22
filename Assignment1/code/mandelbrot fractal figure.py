# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:24:17 2021

@author: Yunxiang
"""

#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# adapted from http://numba.pydata.org/numba-doc/0.35.0/user/examples.html

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




