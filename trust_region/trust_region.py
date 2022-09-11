import numpy as np
from scipy import optimize as op
import math


def trust_region_dog_leg(func, jac, hess, x0, initial_trust_radius = 1.0,
							max_trust_radius = 100, eta = 0.15, gtol = 1e-4, max_iter = 1000):
	xk = x0
	trust_radius = initial_trust_radius
	it = 0
	while True:
		gk = jac(xk)
		Bk = hess(xk)
		Hk = np.linalg.inv(Bk)

		pk = dog_leg(Hk, gk, Bk, trust_radius)

		actual_reduction = func(xk) - func(xk + pk)
		predicted_reduction = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk, pk)))

		if predicted_reduction == 0:
			rhok = 1e99
		else:
			rhok = actual_reduction/predicted_reduction

		norm_pk = math.sqrt(np.dot(pk, pk))


		if rhok < 0.25:
			trust_radius = 0.25* norm_pk
		else:
			if rhok > 0.75 and norm_pk == trust_radius:
				trust_radius = min(2*trust_radius, max_trust_radius)
			else:
				trust_radius = trust_radius

		if rhok > eta:
			xk = xk + pk
		else:
			xk = xk

		if np.linalg.norm(gk) < gtol:
			break
		if it >= max_iter:
			break
		it += 1
	return xk

def dog_leg(Hk, gk, Bk, trust_radius):
	pB = -np.dot(Hk, gk)
	norm_pB = math.sqrt(np.dot(pB, pB))
	if norm_pB <= trust_radius:
		return pB

	pU = - (np.dot(gk, gk) / np.dot(gk, np.dot(Bk, gk))) * gk
	dot_pU = np.dot(pU, pU)
	norm_pU = math.sqrt(dot_pU)
	if norm_pU >= trust_radius:
		return trust_radius * pU / norm_pU

	pB_pU = pB - pU
	dot_pB_pU = np.dot(pB_pU, pB_pU)
	dot_pU_pB_pU = np.dot(pU, pB_pU)
	fact = dot_pU_pB_pU ** 2 - dot_pB_pU * (dot_pU - trust_radius ** 2)
	tau = (-dot_pU_pB_pU + math.sqrt(fact)) / dot_pB_pU

	return pU + tau * (pB- pU)


