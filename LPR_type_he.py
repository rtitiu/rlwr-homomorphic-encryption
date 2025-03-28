from random import SystemRandom
import numpy as np
from math import log2, ceil
from time import time

from poly_multiplication import fast_poly_mult, mod_vec

N = 2 ** 12 
r = 2 ** (4 + 110)
q = r // 2 ** 4
p = q // 2 ** 4

t = 256

Delta = p // t

def uniform_vector(A,B): 
	'''
	Input : integers A,B
	Output: a vector of len N with uniform integer entries in range(A,B+1)  
	'''
	return [SystemRandom().randrange(B - A + 1) + A for _ in range(N)]

def round_vec(vec_x, pp, qq):
	'''
	Input : vec_x a vector of integers
	Output: nearest integer vector to vec_x * pp / qq
	'''
	vec_x = np.array(vec_x)
	return (2 * pp * vec_x + qq) // (2 * qq) 

def int2base(n, b):
	#Input : integer n and a base b
	#Output: a vector of digits corresponding to the decomposition of n in base b     
    if n < b:
        return [n]
    else:
        return [n % b] + int2base(n // b, b) 

def poly_add(p1, p2, modulus = None): 
	'''
	Input : np.arrays p1 and p2 of the same length
	Output: component-wise sum of the two vectors (reduced modulo 'modulus')
	'''
	p1 = np.array(p1, dtype = object)
	p2 = np.array(p2, dtype = object)
	addition = p1 + p2 
	if modulus == None:
		return addition
	else:	
		return np.array([x % modulus for x in addition], dtype = object)	

def poly_mul(p1, p2):
	'''
	Input : integer vectors p1, p2 representing the coefficients of polynomials of degree at most N - 1
	Output: integer vector representing the product polynoial p1 * p2 reduced modulo X^N + 1  
	'''
	p1 = np.array(p1, dtype = object)
	p2 = np.array(p2, dtype = object)
	product = fast_poly_mult(p1, p2)
	product = np.concatenate( (product, [0] * ( 2 * N - len(product) )), None) 
	return np.array([int(product[i]) - int(product[i + N]) for i in range(N)], dtype = object)


def KeyGen():
	sk = uniform_vector(-1,1)
	a = uniform_vector(0,r - 1)
	b = round_vec(poly_mul(a, sk), q, r) % q 
	pk = (a, b)
	return (sk, pk)

def RelinKeyGen(sk, base):
	RelinKey = []
	k = ceil((2 * log2(q) - log2(p)) / log2(base)) #k = ceil(log_base(q ** 2 / p))
	ss = poly_mul(sk, sk)
	for i in range(k):
		v = uniform_vector(0, q - 1) #this is the most expensive computation in ReliNkG
		mask = round_vec(poly_mul(v, sk), p, q) % p
		w = (mask + round_vec(base ** i * ss, p ** 2, q ** 2)) % p
		RelinKey.append((v,w))
	return RelinKey

def Encrypt(message, pub_key):
	(a,b) = pub_key
	rnd = uniform_vector(-1,1)
	c0 = round_vec(poly_mul(a, rnd), q, r) % q
	c1 = round_vec(poly_mul(b, rnd), p, q) % p
	encoded_message = Delta * np.array(message, dtype = object)
	c1 = poly_add(c1, encoded_message) % p
	return (c0,c1)

def Decrypt(ct, sk):
	(c0,c1) = ct
	c0 = np.array(c0, dtype = object)
	c1 = np.array(c1, dtype = object)
	sk = np.array(sk, dtype = object)
	scaled_c0sk = -p * poly_mul(c0,sk)
	scaled_c1 = q * c1	
	return round_vec(poly_add(scaled_c1,scaled_c0sk), t, p * q) % t

def CiphertextAddition(ct1, ct2):
	return (poly_add(ct1[0], ct2[0], q), poly_add(ct1[1], ct2[1], p))

def CiphertextMultiplication(ct1, ct2, rkey):
	base = 2 
	k = ceil((2 * log2(q) - log2(p)) / log2(base)) 
	c2 = round_vec(poly_mul(ct1[1], ct2[1]), t, p) % p
	c1 = round_vec(poly_add(poly_mul(ct1[0], ct2[1]), poly_mul(ct1[1], ct2[0])), t, p) % q
	c0 = round_vec(poly_mul(ct1[0], ct2[0]), t, p) % (q ** 2 // p)

	decomposed_c0 = np.zeros((N, k), dtype = object)	
	for i in range(N):
		into_base = int2base(c0[i],base)
		decomposed_c0[i] = np.array(into_base + [0] * (k - len(into_base)), dtype = object)

	v = c1
	w = c2 
	for j in range(k):

		v = (v + poly_mul(rkey[j][0], decomposed_c0[:,j])) % q
		w = (w + poly_mul(rkey[j][1], decomposed_c0[:,j])) % p

	return (v,w)	

def noise(ct, sk, msg): 
	'''
	Input : ciphertext ct, secret key sk and the message m that ct decrypts to: i.e. Decrypt(ct, sk) == m;
	Output: the function returns pq * noise, where |noise| << 0.5; noise is actually the decryption noise and is of the form = integer / (p * q) 
			More precisely, the output is computed based on the decryption equation t/p * (-p/q * ct[0] * s + ct[1]) = msg + noise

	This function is used in noise_LPR.py script to compute the actual noise values in ciphertexts compared to the theoretical bounds from the paper.		
	'''
	(c0,c1) = ct
	msg = np.array(msg, dtype = object)
	c0 = np.array(c0, dtype = object)
	c1 = np.array(c1, dtype = object)
	if np.array_equal(msg, Decrypt((c0,c1), sk)):
		pqnoise = poly_add(q * t * c1, - p * t * poly_mul(c0,sk))
		pqnoise = poly_add(pqnoise, - p * q * msg)		
		pqnoise = mod_vec(pqnoise, p * q)
		return np.array(pqnoise, dtype = object)	
	else:
		print("Noise is too large! ct does not decrypt correctly!")

if __name__ == '__main__':

	print("N = {}, q = {}, t = {}".format(N, q, t))
	t0 = time()	
	(sk, pk) = KeyGen()
	t1 = time()
	print("KeyGen:        {:.2f}s".format(t1-t0)) 

	t11 = time()
	rkey = RelinKeyGen(sk, 2)
	t2 = time()
	print("Relin Keygen:  {:.2f}s".format(t2 - t11))

	msg1 = uniform_vector(-t//2, t - t//2 -1)
	msg2 = uniform_vector(-t//2, t - t//2 -1)

	t3 = time()
	ct1 = Encrypt(msg1, pk)
	t4 = time()
	print("Encryption:    {:.2f}s".format(t4 - t3))
	ct2 = Encrypt(msg2, pk)

	t9 = time()
	Decrypt(ct1,sk)
	t10 = time()
	print("Decryption:    {:.2f}s".format(t10 - t9))

	t5 = time()
	ct1_plus_ct2 = CiphertextAddition(ct1, ct2)
	t6 = time()
	print("Ct_addition:   {:.2f}s".format(t6 - t5))

	t7 = time()
	ct_multiplied = CiphertextMultiplication(ct1, ct2, rkey)
	t8 = time()
	print("Ct_mult:       {:.2f}s (relin time is included)".format(t8 - t7))

	print(np.array_equal(Decrypt(ct_multiplied, sk), poly_mul(msg1, msg2) % t))
