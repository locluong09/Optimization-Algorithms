import numpy as np
from scipy import optimize as op

from trust_region import trust_region_dog_leg


def f(x):
	return x[0] ** 3 + 8 * x[1] ** 3 - 6 * x[0] * x[1] + 5

def jac(x):
	return np.array([3 * x[0] ** 2 - 6 * x[1], 24 * x[1] ** 2 - 6 * x[0]])

def hess(x):
	return np.array([[6 * x[0], -6], [-6, 48 * x[1]]])


result = trust_region_dog_leg(f, jac, hess, [5, 5])
print("Result of trust region dogleg method: {}".format(result))
print("Value of function at a point: {}".format(f(result)))

x = np.array([10.,10])
rmax=2.
r=.25
eta=1./16
tol=1e-5
opts = {'initial_trust_radius':r, 'max_trust_radius':rmax, 'eta':eta, 'gtol':tol}
sol1 = op.minimize(op.rosen, x, method='dogleg', jac=op.rosen_der, hess=op.rosen_hess, options=opts)
print(sol1)
sol2 = trust_region_dog_leg(op.rosen, op.rosen_der, op.rosen_hess, x, r, rmax, eta = eta, gtol=tol)
print(sol2)
print(np.allclose(sol1.x, sol2))
