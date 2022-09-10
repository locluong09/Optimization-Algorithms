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
ratio = 5

R,C = weymouth(SG, Tav, Zav, L, D)
# print(C)
# print(C.shape)

def residual(x, C, Cwr, P_shut, n, ratio):

	a = np.zeros(14,)

	a[0] = C[2]*mod_sign(x[3] - x[0])*abs(x[3]**2- x[0]**2)**n +\
			C[3]*mod_sign(x[1] - x[0])*abs(x[1]**2- x[0]**2)**n -\
			C[12]*mod_sign(x[0] - x[13])*abs(x[0]**2- x[13]**2)**n

	a[1] = C[9]*mod_sign(x[10] - x[1])*abs(x[10]**2- x[1]**2)**n + \
			C[4]*mod_sign(x[2] - x[1])*abs(x[2]**2- x[1]**2)**n - \
			C[3]*mod_sign(x[1] - x[0])*abs(x[1]**2- x[0]**2)**n

	a[2] = C[10]*mod_sign(x[11] - x[2])*abs(x[11]**2- x[2]**2)**n +\
			C[11]*mod_sign(x[12] - x[2])*abs(x[12]**2- x[2]**2)**n -\
			C[4]*mod_sign(x[2] - x[1])*abs(x[2]**2- x[1]**2)**n

	a[3] = C[8]*mod_sign(x[9] - x[3])*abs(x[9]**2- x[3]**2)**n +\
			C[1]*mod_sign(x[4] - x[3])*abs(x[4]**2- x[3]**2)**n - \
			C[2]*mod_sign(x[3] - x[0])*abs(x[3]**2- x[0]**2)**n

	a[4] = C[7]*mod_sign(x[8] - x[4])*abs(x[8]**2- x[4]**2)**n + \
			C[0]*mod_sign(x[5] - x[4])*abs(x[5]**2- x[4]**2)**n - \
			C[1]*mod_sign(x[4] - x[3])*abs(x[4]**2- x[3]**2)**n

	a[5] = C[5]*mod_sign(x[6] - x[5])*abs(x[6]**2- x[5]**2)**n + \
			C[6]*mod_sign(x[7] - x[5])*abs(x[7]**2- x[5]**2)**n - \
			C[0]*mod_sign(x[5] - x[4])*abs(x[5]**2- x[4]**2)**n

	a[6] = Cwr[0]*mod_sign(P_shut[0] - x[6])*abs(P_shut[0]**2 - x[6]**2)**0.5 -\
			C[5]*mod_sign(x[6] - x[5])*abs(x[6]**2- x[5]**2)**n

	a[7] = Cwr[1]*mod_sign(P_shut[1] - x[7])*abs(P_shut[1]**2 - x[7]**2)**0.5 -\
			C[6]*mod_sign(x[7] - x[5])*abs(x[7]**2- x[5]**2)**n

	a[8] = Cwr[2]*mod_sign(P_shut[2] - x[8])*abs(P_shut[2]**2 - x[8]**2)**0.5 -\
			C[7]*mod_sign(x[8] - x[4])*abs(x[8]**2- x[4]**2)**n

	a[9] = Cwr[3]*mod_sign(P_shut[3] - x[9])*abs(P_shut[3]**2 - x[9]**2)**0.5 -\
			C[8]*mod_sign(x[9] - x[3])*abs(x[9]**2- x[3]**2)**n

	a[10] = Cwr[4]*mod_sign(P_shut[4] - x[10])*abs(P_shut[4]**2 - x[10]**2)**0.5 -\
			C[9]*mod_sign(x[10] - x[1])*abs(x[10]**2- x[1]**2)**n

	a[11] = Cwr[5]*mod_sign(P_shut[5] - x[11])*abs(P_shut[5]**2 - x[11]**2)**0.5 -\
			C[10]*mod_sign(x[11] - x[2])*abs(x[11]**2- x[2]**2)**n
	
	a[12] = Cwr[6]*mod_sign(P_shut[6] - x[12])*abs(P_shut[6]**2 - x[12]**2)**0.5 -\
			C[11]*mod_sign(x[12] - x[2])*abs(x[12]**2- x[2]**2)**n

	a[13] = x[13] - 1200/ratio
	return a

def Jacobian(x, C, Cwr, P_shut, n):
	a = np.zeros((14,14))

	a[0,0] = -C[2]*mod_sign(x[3] - x[0])*2*n*abs(x[0])*abs(x[3]**2- x[0]**2)**(n-1) +\
			-C[3]*mod_sign(x[1] - x[0])*2*n*abs(x[0])*abs(x[1]**2- x[0]**2)**(n-1) -\
			C[12]*mod_sign(x[0] - x[13])*2*n*abs(x[0])*abs(x[0]**2- x[13]**2)**(n-1)
	a[0,1] = C[3]*mod_sign(x[1] - x[0])*2*n*abs(x[1])*abs(x[1]**2- x[0]**2)**(n-1)
	a[0,3] = C[2]*mod_sign(x[3] - x[0])*2*n*abs(x[3])*abs(x[3]**2- x[0]**2)**(n-1)
	a[0,13] = C[12]*mod_sign(x[0] - x[13])*2*n*abs(x[13])*abs(x[0]**2- x[13]**2)**(n-1)


	a[1,0] = C[3]*mod_sign(x[1] - x[0])*2*n*abs(x[0])*abs(x[1]**2- x[0]**2)**(n-1)
	a[1,1] = -C[9]*mod_sign(x[10] - x[1])*2*n*abs(x[1])*abs(x[10]**2- x[1]**2)**(n-1) + \
			-C[4]*mod_sign(x[2] - x[1])*2*n*abs(x[1])*abs(x[2]**2- x[1]**2)**(n-1) - \
			C[3]*mod_sign(x[1] - x[0])*2*n*abs(x[1])*abs(x[1]**2- x[0]**2)**(n-1)
	a[1,2] = C[4]*mod_sign(x[2] - x[1])*2*n*abs(x[2])*abs(x[2]**2- x[1]**2)**(n-1)
	a[1,10] = C[9]*mod_sign(x[10] - x[1])*2*n*abs(x[10])*abs(x[10]**2- x[1]**2)**(n-1)


	a[2,1] = C[4]*mod_sign(x[2] - x[1])*2*n*abs(x[1])*abs(x[2]**2- x[1]**2)**(n-1)
	a[2,2] = -C[10]*mod_sign(x[11] - x[2])*2*n*abs(x[2])*abs(x[11]**2- x[2]**2)**(n-1) +\
			-C[11]*mod_sign(x[12] - x[2])*2*n*abs(x[2])*abs(x[12]**2- x[2]**2)**(n-1) -\
			C[4]*mod_sign(x[2] - x[1])*2*n*abs(x[2])*abs(x[2]**2- x[1]**2)**(n-1)
	a[2,11] = C[10]*mod_sign(x[11] - x[2])*2*n*abs(x[11])*abs(x[11]**2- x[2]**2)**(n-1)
	a[2,12] = C[11]*mod_sign(x[12] - x[2])*2*n*abs(x[12])*abs(x[12]**2- x[2]**2)**(n-1)


	a[3,0] = C[2]*mod_sign(x[3] - x[0])*2*n*abs(x[0])*abs(x[3]**2- x[0]**2)**(n-1)
	a[3,3] = -C[8]*mod_sign(x[9] - x[3])*2*n*abs(x[3])*abs(x[9]**2- x[3]**2)**(n-1) +\
			-C[1]*mod_sign(x[4] - x[3])*2*n*abs(x[3])*abs(x[4]**2- x[3]**2)**(n-1) - \
			C[2]*mod_sign(x[3] - x[0])*2*n*abs(x[3])*abs(x[3]**2- x[0]**2)**(n-1)
	a[3,4] = C[1]*mod_sign(x[4] - x[3])*2*n*abs(x[4])*abs(x[4]**2- x[3]**2)**(n-1)
	a[3,9] = C[8]*mod_sign(x[9] - x[3])*2*n*abs(x[9])*abs(x[9]**2- x[3]**2)**(n-1) 


	a[4,3] = C[1]*mod_sign(x[4] - x[3])*2*n*abs(x[3])*abs(x[4]**2- x[3]**2)**(n-1)
	a[4,4] = -C[7]*mod_sign(x[8] - x[4])*2*n*abs(x[4])*abs(x[8]**2- x[4]**2)**(n-1) + \
			-C[0]*mod_sign(x[5] - x[4])*2*n*abs(x[4])*abs(x[5]**2- x[4]**2)**(n-1) - \
			C[1]*mod_sign(x[4] - x[3])*2*n*abs(x[4])*abs(x[4]**2- x[3]**2)**(n-1)
	a[4,5] = C[0]*mod_sign(x[5] - x[4])*2*n*abs(x[5])*abs(x[5]**2- x[4]**2)**(n-1) 
	a[4,8] = C[7]*mod_sign(x[8] - x[4])*2*n*abs(x[8])*abs(x[8]**2- x[4]**2)**(n-1)


	a[5,4] = C[0]*mod_sign(x[5] - x[4])*2*n*abs(x[4])*abs(x[5]**2- x[4]**2)**(n-1)
	a[5,5] = -C[5]*mod_sign(x[6] - x[5])*2*n*abs(x[5])*abs(x[6]**2- x[5]**2)**(n-1) + \
			-C[6]*mod_sign(x[7] - x[5])*2*n*abs(x[5])*abs(x[7]**2- x[5]**2)**(n-1) - \
			C[0]*mod_sign(x[5] - x[4])*2*n*abs(x[5])*abs(x[5]**2- x[4]**2)**(n-1)
	a[5,6] = C[5]*mod_sign(x[6] - x[5])*2*n*abs(x[6])*abs(x[6]**2- x[5]**2)**(n-1)
	a[5,7] = C[6]*mod_sign(x[7] - x[5])*2*n*abs(x[7])*abs(x[7]**2- x[5]**2)**(n-1) 


	a[6,5] = C[5]*mod_sign(x[6] - x[5])*2*n*abs(x[5])*abs(x[6]**2- x[5]**2)**(n-1)
	a[6,6] = -Cwr[0]*mod_sign(P_shut[0] - x[6])*abs(x[6])*abs(P_shut[0]**2 - x[6]**2)**-0.5 -\
			C[5]*mod_sign(x[6] - x[5])*2*n*abs(x[6])*abs(x[6]**2- x[5]**2)**(n-1)


	a[7,5] = C[6]*mod_sign(x[7] - x[5])*2*n*abs(x[5])*abs(x[7]**2- x[5]**2)**(n-1)
	a[7,7] = -Cwr[1]*mod_sign(P_shut[1] - x[7])*abs(x[7])*abs(P_shut[1]**2 - x[7]**2)**-0.5 -\
			C[6]*mod_sign(x[7] - x[5])*2*n*abs(x[7])*abs(x[7]**2- x[5]**2)**(n-1)


	a[8,4] = C[7]*mod_sign(x[8] - x[4])*2*n*abs(x[4])*abs(x[8]**2- x[4]**2)**(n-1)
	a[8,8] = -Cwr[2]*mod_sign(P_shut[2] - x[8])*abs(x[8])*abs(P_shut[2]**2 - x[8]**2)**-0.5 -\
			C[7]*mod_sign(x[8] - x[4])*2*n*abs(x[8])*abs(x[8]**2- x[4]**2)**(n-1)


	a[9,3] = C[8]*mod_sign(x[9] - x[3])*2*n*abs(x[3])*abs(x[9]**2- x[3]**2)**(n-1)
	a[9,9] = -Cwr[3]*mod_sign(P_shut[3] - x[9])*abs(x[9])*abs(P_shut[3]**2 - x[9]**2)**-0.5 -\
			C[8]*mod_sign(x[9] - x[3])*2*n*abs(x[9])*abs(x[9]**2- x[3]**2)**(n-1)


	a[10,1] = 	C[9]*mod_sign(x[10] - x[1])*2*n*abs(x[1])*abs(x[10]**2- x[1]**2)**(n-1)
	a[10,10] = -Cwr[4]*mod_sign(P_shut[4] - x[10])*abs(x[10])*abs(P_shut[4]**2 - x[10]**2)**-0.5 -\
			C[9]*mod_sign(x[10] - x[1])*2*n*abs(x[10])*abs(x[10]**2- x[1]**2)**(n-1)


	a[11,2] = C[10]*mod_sign(x[11] - x[2])*2*n*abs(x[2])*abs(x[11]**2- x[2]**2)**(n-1)
	a[11,11] = -Cwr[5]*mod_sign(P_shut[5] - x[11])*abs(x[11])*abs(P_shut[5]**2 - x[11]**2)**-0.5 -\
			C[10]*mod_sign(x[11] - x[2])*2*n*abs(x[11])*abs(x[11]**2- x[2]**2)**(n-1)
	

	a[12,2] = C[11]*mod_sign(x[12] - x[2])*2*n*abs(x[2])*abs(x[12]**2- x[2]**2)**(n-1)
	a[12,12] = -Cwr[6]*mod_sign(P_shut[6] - x[12])*abs(x[12])*abs(P_shut[6]**2 - x[12]**2)**-0.5 -\
			C[11]*mod_sign(x[12] - x[2])*2*n*abs(x[12])*abs(x[12]**2- x[2]**2)**(n-1)

	a[13,13] = 1
	return a


def mod_sign(x):
	if x > 0:
		return 1
	else:
		return -1 

x = np.array([755, 780, 810, 770, 800, 820, 860, 870, 874, 865, 855.5, 900, 890, 200])
# print(residual(x, C, Cwr, P_shut, n, ratio))
# J = Jacobian(x, C, Cwr, P_shut, n)
# print(J[12,12])

def f(x):
	return 0.5*(residual(x, C, Cwr, P_shut, n, ratio)**2).sum()

def J(x):
	return Jacobian(x, C, Cwr, P_shut, n)

def g(x):
	return J(x).dot(residual(x, C, Cwr, P_shut, n, ratio))

def H(x):
	return Jacobian(x, C, Cwr, P_shut, n).T.dot(Jacobian(x, C, Cwr, P_shut, n))


rmax=100
r=1
eta=1./16
tol=1e-5

sol2 = trust_region_dog_leg(f, g, H, x, r, rmax, eta = eta, gtol=tol)
print(sol2)


# from math import sin, cos

# # define the system of equations
# def ra(x):
# 	return np.array([-sin(x[0])*cos(x[1]) - 2*cos(x[0])*sin(x[1]), -sin(x[1])*cos(x[0]) - 2*cos(x[1])*sin(x[0])])

# def fa(x):
# 	return .5*(ra(x)**2).sum()
# def Ja(x):
# 	return np.array([[-cos(x[0])*cos(x[1]) + 2*sin(x[0])*sin(x[1]), sin(x[0])*sin(x[1]) - 2*cos(x[0])*cos(x[1])],
# 	[sin(x[1])*sin(x[0]) - 2*cos(x[1])*cos(x[0]), -cos(x[1])*cos(x[0]) + 2*sin(x[1])*sin(x[0])]])

# def ga(x):
# 	return Ja(x).dot(ra(x))

# def Ha(x):
# 	return Ja(x).T.dot(Ja(x))
# rmax=2.
# rr=.25
# eta=1./16
# tol=1e-5

# x = np.array([3.5, -2.5])
# xstar = trust_region_dog_leg(fa,ga,Ha,x,rr,rmax,eta=eta,gtol=tol)
# print(xstar)
# print(ra(xstar))






