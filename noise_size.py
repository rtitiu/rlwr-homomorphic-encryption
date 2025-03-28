'''
noise_size.py gives a comparison of the actual noise observed in this implementation's ciphertexts vs theoretical bounds from the paper.
'''

from LPR_type_he import *
from math import sqrt, log2
from decimal import Decimal
import decimal

decimal.setcontext(decimal.Context(prec = 900))
n = Decimal(N)
q = Decimal(q)
p = q / 16	
epsilon_p = Decimal((int(p) % t) / t)	
N_LPR = t * Decimal.sqrt(4 * n ** 2 + 3 * n * 256 * (1 +  (t * epsilon_p)**2 )) / q
N_LPR_scaled_by_pq = round(p * q * N_LPR) #fresh LPR noise multiplied by p * q

k = Decimal(int(log2(q ** 2 / p)) + 1) #number of RLWR samples that make up the ReLin key
N_relin = n * t * Decimal.sqrt((k+1) * 3/2) / p #we assumed here that c_2 is decomposed in base w = 2.
A = t * Decimal.sqrt(2 * n ** 2 + 3 * n)
B = (n * Decimal.sqrt(3 * n)/4 + Decimal.sqrt(2 * n ** 2 + 3 * 256 * n)) * t / q
N_LPR_mult = N_LPR ** 2 + 2 * A * N_LPR + B + N_relin
N_LPR_mult_scaled_by_pq = round(p * q * N_LPR_mult)

no_trials = 10
(sk, pk) = KeyGen()

fresh_noise_inf_norms = []
sum_noise_inf_norms = []
prod_noise_inf_norms = []

rkey = RelinKeyGen(sk, 2)
for trial in range(no_trials):
	print(trial)
	msg1 =  uniform_vector(0,t-1)
	msg2 =  uniform_vector(0,t-1)
	msg_sum = poly_add(msg1, msg2) % t 
	msg_prod = poly_mul(msg1, msg2) % t

	ct1 = Encrypt(msg1, pk)
	ct2 = Encrypt(msg2, pk)
	ct_sum = CiphertextAddition(ct1, ct2)
	ct_prod = CiphertextMultiplication(ct1,ct2, rkey)

	noise1 = noise(ct1, sk, msg1)
	noise_sum = noise(ct_sum, sk, msg_sum)
	noise_prod = noise(ct_prod, sk, msg_prod)

	fresh_noise_inf_norms.append(max([abs(x) for x in noise1]) / N_LPR_scaled_by_pq )
	sum_noise_inf_norms.append(max([abs(x) for x in noise_sum]) / (2 * N_LPR_scaled_by_pq) )
	prod_noise_inf_norms.append(max([abs(x) for x in noise_prod]) / (N_LPR_mult_scaled_by_pq))

print("N = {}, q = {}, t = {}".format(N,q,t))
print("Mean values over {} trials for:".format(no_trials))
print("fresh ct: |actual_noise|_inf / theoretical_bound : {:.3f}".format(sum(fresh_noise_inf_norms) / no_trials))
print("sum     : |actual_noise|_inf / theoretical_bound : {:.3f}".format(sum(sum_noise_inf_norms) / no_trials))
print("product : |actual_noise|_inf / theoretical_bound : {:.3f}".format(sum(prod_noise_inf_norms) / no_trials))
#-----------------
