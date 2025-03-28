'''
poly_multiplication.py implements a polynomial multiplication leveraging scipy's complex FFT implementation. To multiply degree N polynomials f and g with integer coefficients one calls fast_poly_mult(f,g).
More details on how to use the complex FFT to multiply integer polynomials are given https://github.com/rtitiu/polymul-approx-ffts. 
'''

import numpy as np
from scipy import signal
from random import SystemRandom
from math import log2, ceil
from time import time
import scipy

def mod_vec(x_vec, modulus):
	''' 
	Input : vector of integers x_vec and an integer modulus
	Output: x_vec % modulus with representatives in [-modulus/2, modulus/2)
	'''
	log_modulus = ceil(log2(modulus))
	x_vec = np.array(x_vec, dtype = object)
	positive_r = np.bitwise_and(x_vec, modulus - 1)
	r_vec = positive_r - (modulus * (np.round(np.divide(positive_r, modulus).astype(float)).astype(int)))
	return r_vec

def poly_base_decomposition(f, B): 
	N = len(f)
	f = np.array(f, dtype = object)
	k = ceil( log2(2 * max(f) + 1) / log2(B))
	decomposed_f = np.zeros((N, k), dtype = object)
	for j in range(k):
		r = mod_vec(f, B)
		decomposed_f[:,j] = r 
		f = (f - r) // B
	return decomposed_f

def fast_mult(decomposed_f, decomposed_g, B = 2 ** 19):
	N = decomposed_f.shape[0]
	k_f = decomposed_f.shape[1]
	k_g = decomposed_g.shape[1]
	decomposed_fg = np.zeros((N + 1, k_f + k_g - 1), dtype = 'complex128')	

	for i in range(k_f):
		for j in range(k_g):
			decomposed_fg[:,i + j] += scipy.fft.rfft(decomposed_f[:, i], 2 * N) * scipy.fft.rfft(decomposed_g[:, j], 2 * N) 
	fg_recovered = np.array([0] * (2 * N), dtype = object)
	for j in reversed(range(k_f + k_g - 1)):
		rounded_term = scipy.fft.irfft(decomposed_fg[:,j], 2 * N)
		rounded_term = np.round(rounded_term.real).astype(int)
		fg_recovered *= B
		fg_recovered += rounded_term
	return fg_recovered	
	
def fast_poly_mult(f, g, base = 2 ** 19):
	f = np.array(f, dtype = 'object')
	g = np.array(g, dtype = 'object')
	f_dec = poly_base_decomposition(f, base)	
	g_dec = poly_base_decomposition(g, base)	

	return fast_mult(f_dec, g_dec, base)
