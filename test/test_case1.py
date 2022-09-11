import pytest

import numpy as np
import scipy as sp
from scipy import optimize as op

from trust_region import trust_region_dog_leg

global C, Cwr, P_shut, n, ratio

def weymouth(SG, Tav, Zav, L, D):
	m = 16/3
	n = 0.5
	eff = 1
	Psc = 14.7
	Tsc = 60 + 460
	sigma = 438
	const = SG*Tav*Zav/sigma**2*Psc**2/Tsc**2
	RG = const*np.multiply(L, 1/np.power(D,m))
	CG = eff*1/np.power(RG,n)
	return RG, CG

SG = 0.62
Tav = 82+460
Zav = 0.91
D = np.array([4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4])
L = np.array([4500, 5000, 5500, 5300, 5100, 4600, 4800, 4700, 5500, 5000, 4600, 5400, 8000])/5280
P_shut = np.array([1500, 1300, 880, 1400, 900, 1450, 950])
Cwr = np.array([5.5, 5.2, 4.2, 5.0, 3.5, 5.7, 3.7])*1000
m = 16/3
n = 0.5
ratio = 7

R,C = weymouth(SG, Tav, Zav, L, D)
# print(C)
# print(C.shape)

def residual(x, C, Cwr, P_shut, n, ratio):

	a = np.zeros(13,)

	a[0] = C[2]*mod_sign1(x[3] - x[0])*abs(x[3]**2- x[0]**2)**n +\
			C[3]*mod_sign1(x[1] - x[0])*abs(x[1]**2- x[0]**2)**n -\
			C[12]*mod_sign1(x[0] - 1200/ratio)*abs(x[0]**2- (1200/ratio)**2)**n

	a[1] = C[9]*mod_sign1(x[10] - x[1])*abs(x[10]**2- x[1]**2)**n + \
			C[4]*mod_sign1(x[2] - x[1])*abs(x[2]**2- x[1]**2)**n - \
			C[3]*mod_sign1(x[1] - x[0])*abs(x[1]**2- x[0]**2)**n

	a[2] = C[10]*mod_sign1(x[11] - x[2])*abs(x[11]**2- x[2]**2)**n +\
			C[11]*mod_sign1(x[12] - x[2])*abs(x[12]**2- x[2]**2)**n -\
			C[4]*mod_sign1(x[2] - x[1])*abs(x[2]**2- x[1]**2)**n

	a[3] = C[8]*mod_sign1(x[9] - x[3])*abs(x[9]**2- x[3]**2)**n +\
			C[1]*mod_sign1(x[4] - x[3])*abs(x[4]**2- x[3]**2)**n - \
			C[2]*mod_sign1(x[3] - x[0])*abs(x[3]**2- x[0]**2)**n

	a[4] = C[7]*mod_sign1(x[8] - x[4])*abs(x[8]**2- x[4]**2)**n + \
			C[0]*mod_sign1(x[5] - x[4])*abs(x[5]**2- x[4]**2)**n - \
			C[1]*mod_sign1(x[4] - x[3])*abs(x[4]**2- x[3]**2)**n

	a[5] = C[5]*mod_sign1(x[6] - x[5])*abs(x[6]**2- x[5]**2)**n + \
			C[6]*mod_sign1(x[7] - x[5])*abs(x[7]**2- x[5]**2)**n - \
			C[0]*mod_sign1(x[5] - x[4])*abs(x[5]**2- x[4]**2)**n

	a[6] = Cwr[0]*mod_sign1(P_shut[0] - x[6])*abs(P_shut[0]**2 - x[6]**2)**0.5 -\
			C[5]*mod_sign1(x[6] - x[5])*abs(x[6]**2- x[5]**2)**n

	a[7] = Cwr[1]*mod_sign1(P_shut[1] - x[7])*abs(P_shut[1]**2 - x[7]**2)**0.5 -\
			C[6]*mod_sign1(x[7] - x[5])*abs(x[7]**2- x[5]**2)**n

	a[8] = Cwr[2]*mod_sign1(P_shut[2] - x[8])*abs(P_shut[2]**2 - x[8]**2)**0.5 -\
			C[7]*mod_sign1(x[8] - x[4])*abs(x[8]**2- x[4]**2)**n

	a[9] = Cwr[3]*mod_sign1(P_shut[3] - x[9])*abs(P_shut[3]**2 - x[9]**2)**0.5 -\
			C[8]*mod_sign1(x[9] - x[3])*abs(x[9]**2- x[3]**2)**n

	a[10] = Cwr[4]*mod_sign1(P_shut[4] - x[10])*abs(P_shut[4]**2 - x[10]**2)**0.5 -\
			C[9]*mod_sign1(x[10] - x[1])*abs(x[10]**2- x[1]**2)**n

	a[11] = Cwr[5]*mod_sign1(P_shut[5] - x[11])*abs(P_shut[5]**2 - x[11]**2)**0.5 -\
			C[10]*mod_sign1(x[11] - x[2])*abs(x[11]**2- x[2]**2)**n
	
	a[12] = Cwr[6]*mod_sign1(P_shut[6] - x[12])*abs(P_shut[6]**2 - x[12]**2)**0.5 -\
			C[11]*mod_sign1(x[12] - x[2])*abs(x[12]**2- x[2]**2)**n

	
	return a

def Jacobian(x, C, Cwr, P_shut, n):
	a = np.zeros((13,13))

	a[0,0] = np.sign(x[0] - x[3])*C[2]*mod_sign(x[3] - x[0])*2*n*(x[0])*abs(x[3]**2- x[0]**2)**(n-1) +\
			np.sign(x[0] - x[1])*C[3]*mod_sign(x[1] - x[0])*2*n*(x[0])*abs(x[1]**2- x[0]**2)**(n-1) -\
			np.sign(x[0] - 1200/ratio)*C[12]*mod_sign(x[0] - 1200/ratio)*2*n*(x[0])*abs(x[0]**2- (1200/ratio)**2)**(n-1)
	a[0,1] = np.sign(x[1] - x[0])*C[3]*mod_sign(x[1] - x[0])*2*n*(x[1])*abs(x[1]**2- x[0]**2)**(n-1)
	a[0,3] = np.sign(x[3] - x[0])*C[2]*mod_sign(x[3] - x[0])*2*n*(x[3])*abs(x[3]**2- x[0]**2)**(n-1)
	# a[0,13] = -np.sign(1200/ratio - x[0])*C[12]*mod_sign(x[0] - 1200/ratio)*2*n*(1200/ratio)*abs(x[0]**2- 1200/ratio**2)**(n-1)


	a[1,0] = -np.sign(x[0] - x[1])*C[3]*mod_sign(x[1] - x[0])*2*n*(x[0])*abs(x[1]**2- x[0]**2)**(n-1)
	a[1,1] = np.sign(x[1] - x[10])*C[9]*mod_sign(x[10] - x[1])*2*n*(x[1])*abs(x[10]**2- x[1]**2)**(n-1) + \
			np.sign(x[1] - x[2])*C[4]*mod_sign(x[2] - x[1])*2*n*(x[1])*abs(x[2]**2- x[1]**2)**(n-1) - \
			np.sign(x[1] - x[0])*C[3]*mod_sign(x[1] - x[0])*2*n*(x[1])*abs(x[1]**2- x[0]**2)**(n-1)
	a[1,2] = np.sign(x[2] - x[1])*C[4]*mod_sign(x[2] - x[1])*2*n*(x[2])*abs(x[2]**2- x[1]**2)**(n-1)
	a[1,10] = np.sign(x[10] - x[1])*C[9]*mod_sign(x[10] - x[1])*2*n*(x[10])*abs(x[10]**2- x[1]**2)**(n-1)


	a[2,1] = -np.sign(x[1] - x[2])*C[4]*mod_sign(x[2] - x[1])*2*n*(x[1])*abs(x[2]**2- x[1]**2)**(n-1)
	a[2,2] = np.sign(x[2] - x[11])*C[10]*mod_sign(x[11] - x[2])*2*n*(x[2])*abs(x[11]**2- x[2]**2)**(n-1) +\
			np.sign(x[2] - x[12])*C[11]*mod_sign(x[12] - x[2])*2*n*(x[2])*abs(x[12]**2- x[2]**2)**(n-1) -\
			np.sign(x[2] - x[1])*C[4]*mod_sign(x[2] - x[1])*2*n*(x[2])*abs(x[2]**2- x[1]**2)**(n-1)
	a[2,11] = np.sign(x[11] - x[2])*C[10]*mod_sign(x[11] - x[2])*2*n*(x[11])*abs(x[11]**2- x[2]**2)**(n-1)
	a[2,12] = np.sign(x[12] - x[2])*C[11]*mod_sign(x[12] - x[2])*2*n*(x[12])*abs(x[12]**2- x[2]**2)**(n-1)


	a[3,0] = -np.sign(x[0] - x[3])*C[2]*mod_sign(x[3] - x[0])*2*n*(x[0])*abs(x[3]**2- x[0]**2)**(n-1)
	a[3,3] = np.sign(x[3] - x[9])*C[8]*mod_sign(x[9] - x[3])*2*n*(x[3])*abs(x[9]**2- x[3]**2)**(n-1) +\
			np.sign(x[3] - x[4])*C[1]*mod_sign(x[4] - x[3])*2*n*(x[3])*abs(x[4]**2- x[3]**2)**(n-1) - \
			np.sign(x[3] - x[0])*C[2]*mod_sign(x[3] - x[0])*2*n*(x[3])*abs(x[3]**2- x[0]**2)**(n-1)
	a[3,4] = np.sign(x[4] - x[3])*C[1]*mod_sign(x[4] - x[3])*2*n*(x[4])*abs(x[4]**2- x[3]**2)**(n-1)
	a[3,9] = np.sign(x[9] - x[3])*C[8]*mod_sign(x[9] - x[3])*2*n*(x[9])*abs(x[9]**2- x[3]**2)**(n-1) 


	a[4,3] = -np.sign(x[3] - x[4])*C[1]*mod_sign(x[4] - x[3])*2*n*(x[3])*abs(x[4]**2- x[3]**2)**(n-1)
	a[4,4] = np.sign(x[4] - x[8])*C[7]*mod_sign(x[8] - x[4])*2*n*(x[4])*abs(x[8]**2- x[4]**2)**(n-1) + \
			np.sign(x[4] - x[5])*C[0]*mod_sign(x[5] - x[4])*2*n*(x[4])*abs(x[5]**2- x[4]**2)**(n-1) - \
			np.sign(x[4] - x[3])*C[1]*mod_sign(x[4] - x[3])*2*n*(x[4])*abs(x[4]**2- x[3]**2)**(n-1)
	a[4,5] = np.sign(x[5] - x[4])*C[0]*mod_sign(x[5] - x[4])*2*n*(x[5])*abs(x[5]**2- x[4]**2)**(n-1) 
	a[4,8] = np.sign(x[8] - x[4])*C[7]*mod_sign(x[8] - x[4])*2*n*(x[8])*abs(x[8]**2- x[4]**2)**(n-1)


	a[5,4] = -np.sign(x[4] - x[5])*C[0]*mod_sign(x[5] - x[4])*2*n*(x[4])*abs(x[5]**2- x[4]**2)**(n-1)
	a[5,5] = np.sign(x[5] - x[6])*C[5]*mod_sign(x[6] - x[5])*2*n*(x[5])*abs(x[6]**2- x[5]**2)**(n-1) + \
			np.sign(x[5] - x[7])*C[6]*mod_sign(x[7] - x[5])*2*n*(x[5])*abs(x[7]**2- x[5]**2)**(n-1) - \
			np.sign(x[5] - x[4])*C[0]*mod_sign(x[5] - x[4])*2*n*(x[5])*abs(x[5]**2- x[4]**2)**(n-1)
	a[5,6] = np.sign(x[6] - x[5])*C[5]*mod_sign(x[6] - x[5])*2*n*(x[6])*abs(x[6]**2- x[5]**2)**(n-1)
	a[5,7] = np.sign(x[7] - x[5])*C[6]*mod_sign(x[7] - x[5])*2*n*(x[7])*abs(x[7]**2- x[5]**2)**(n-1) 


	a[6,5] = -np.sign(x[5] - x[6])*C[5]*mod_sign(x[6] - x[5])*2*n*(x[5])*abs(x[6]**2- x[5]**2)**(n-1)
	a[6,6] = np.sign(x[6] - P_shut[0])*Cwr[0]*mod_sign(P_shut[0] - x[6])*(x[6])*abs(P_shut[0]**2 - x[6]**2)**-0.5 -\
			np.sign(x[6] - x[5])*C[5]*mod_sign(x[6] - x[5])*2*n*(x[6])*abs(x[6]**2- x[5]**2)**(n-1)


	a[7,5] = -np.sign(x[6] - x[7])*C[6]*mod_sign(x[7] - x[5])*2*n*(x[5])*abs(x[7]**2- x[5]**2)**(n-1)
	a[7,7] = np.sign(x[7] - P_shut[1])*Cwr[1]*mod_sign(P_shut[1] - x[7])*(x[7])*abs(P_shut[1]**2 - x[7]**2)**-0.5 -\
			np.sign(x[7] - x[6])*C[6]*mod_sign(x[7] - x[5])*2*n*(x[7])*abs(x[7]**2- x[5]**2)**(n-1)


	a[8,4] = -np.sign(x[4] - x[8])*C[7]*mod_sign(x[8] - x[4])*2*n*(x[4])*abs(x[8]**2- x[4]**2)**(n-1)
	a[8,8] = np.sign(x[8] - P_shut[2])*Cwr[2]*mod_sign(P_shut[2] - x[8])*(x[8])*abs(P_shut[2]**2 - x[8]**2)**-0.5 -\
			np.sign(x[8] - x[4])*C[7]*mod_sign(x[8] - x[4])*2*n*(x[8])*abs(x[8]**2- x[4]**2)**(n-1)


	a[9,3] = -np.sign(x[3] - x[9])*C[8]*mod_sign(x[9] - x[3])*2*n*(x[3])*abs(x[9]**2- x[3]**2)**(n-1)
	a[9,9] = np.sign(x[9] - P_shut[3])*Cwr[3]*mod_sign(P_shut[3] - x[9])*(x[9])*abs(P_shut[3]**2 - x[9]**2)**-0.5 -\
			np.sign(x[9] - x[3])*C[8]*mod_sign(x[9] - x[3])*2*n*(x[9])*abs(x[9]**2- x[3]**2)**(n-1)


	a[10,1] = 	-np.sign(x[1] - x[10])*C[9]*mod_sign(x[10] - x[1])*2*n*(x[1])*abs(x[10]**2- x[1]**2)**(n-1)
	a[10,10] = np.sign(x[10] - P_shut[4])*Cwr[4]*mod_sign(P_shut[4] - x[10])*(x[10])*abs(P_shut[4]**2 - x[10]**2)**-0.5 -\
			np.sign(x[10] - x[1])*C[9]*mod_sign(x[10] - x[1])*2*n*(x[10])*abs(x[10]**2- x[1]**2)**(n-1)


	a[11,2] = -np.sign(x[2] - x[11])*C[10]*mod_sign(x[11] - x[2])*2*n*(x[2])*abs(x[11]**2- x[2]**2)**(n-1)
	a[11,11] = np.sign(x[11] - P_shut[5])*Cwr[5]*mod_sign(P_shut[5] - x[11])*(x[11])*abs(P_shut[5]**2 - x[11]**2)**-0.5 -\
			np.sign(x[11] - x[2])*C[10]*mod_sign(x[11] - x[2])*2*n*(x[11])*abs(x[11]**2- x[2]**2)**(n-1)
	

	a[12,2] = np.sign(x[2] - x[12])*C[11]*mod_sign(x[12] - x[2])*2*n*(x[2])*abs(x[12]**2- x[2]**2)**(n-1)
	a[12,12] = np.sign(x[12] - P_shut[6])*Cwr[6]*mod_sign(P_shut[6] - x[12])*(x[12])*abs(P_shut[6]**2 - x[12]**2)**-0.5 -\
			np.sign(x[12] - x[2])*C[11]*mod_sign(x[12] - x[2])*2*n*(x[12])*abs(x[12]**2- x[2]**2)**(n-1)

	return a


def mod_sign(x):
	if x > 0:
		return 1
	else:
		return -1 

def mod_sign1(x):
	if x > 0:
		return 1
	else:
		return -1

print(C)
x = np.array([755, 780, 810, 770, 800, 820, 860, 870, 874, 865, 855.5, 900, 890])
# x = np.array([746.439558793830,
# 778.938437828739,
# 799.752696014779,
# 835.593794641774,
# 870.702615290880,
# 898.281144248301,
# 1209.36784920293,
# 1092.22238651679,
# 873.796606671877,
# 1125.95416307974,
# 813.179142915890,
# 1153.36285681795,
# 848.665490787991,
# 171.428571428571])
print(residual(x, C, Cwr, P_shut, n, ratio))


while True:
    J = Jacobian(x, C, Cwr, P_shut, n)
    res = residual(x, C, Cwr, P_shut, n, ratio)
    delta = np.linalg.inv(J).dot(res)
    x = x - delta
    if np.linalg.norm(delta, np.inf) < 0.01:
        break
print(x)

