from math import sin, cos
import numpy as np

from trust_region import trust_region_dog_leg

# define the system of equations
def ra(x):
	return np.array([-sin(x[0])*cos(x[1]) - 2*cos(x[0])*sin(x[1]), -sin(x[1])*cos(x[0]) - 2*cos(x[1])*sin(x[0])])

def fa(x):
	return .5*(ra(x)**2).sum()
def Ja(x):
	return np.array([[-cos(x[0])*cos(x[1]) + 2*sin(x[0])*sin(x[1]), sin(x[0])*sin(x[1]) - 2*cos(x[0])*cos(x[1])],
	[sin(x[1])*sin(x[0]) - 2*cos(x[1])*cos(x[0]), -cos(x[1])*cos(x[0]) + 2*sin(x[1])*sin(x[0])]])

def ga(x):
	return Ja(x).dot(ra(x))

def Ha(x):
	return Ja(x).T.dot(Ja(x))
rmax=2.
rr=.25
eta=1./16
tol=1e-5

x = np.array([3.5, -2.5])
xstar = trust_region_dog_leg(fa,ga,Ha,x,rr,rmax,eta=eta,gtol=tol)
print(xstar)
print(ra(xstar))