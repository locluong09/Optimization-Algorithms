# Determine pressure drop in natural gas network using Q_formula

import numpy as np
from trust_region import trust_region_dog_leg

global C1, C2, C3, C4, C5

gamma = 0.7
Tav = 70 +460
Zav = 0.9
Tsc = 60 + 460
Psc = 14.7
D = [6,4,4,4,6]
L = [6,7,2,2,3]
gc = 32.17
R = 10.731
MW = 28.966
S = 4650
D1 = 1500
D2 = 3150
m = 16/3
n = 0.5


sigma = 433.5
alpha = 1
rG = gamma*Tav*Zav/sigma**2*Psc**2/Tsc**2
R1 = alpha*rG*L[0]/D[0]**m
R2 = alpha*rG*L[1]/D[1]**m
R3 = alpha*rG*L[2]/D[2]**m
R4 = alpha*rG*L[3]/D[3]**m
R5 = alpha*rG*L[4]/D[4]**m


C1 = 1/R1**n
C2 = 1/R2**n
C3 = 1/R3**n
C4 = 1/R4**n
C5 = 1/R5**n

x = np.array([500,150,300]).reshape(-1,)
x = np.array([1,2,3]).reshape(-1,)


def residual(x,C1,C2,C3,C4,C5,n = 0.5):
    a = np.zeros(3,)
    a[0] = C3*np.sign(x[2] - x[0])*abs(x[2]**2- x[0]**2)**n - \
        C1*np.sign(x[0] - 200)*abs(x[0]**2- 200**2)**n + 4650*1e3
    a[1] = C4*np.sign(200 - x[1])*abs(200**2- x[1]**2)**n -\
        C5*np.sign(x[1] - x[2])*abs(x[1]**2- x[2]**2)**n - 3150*1e3
    a[2] = C5*np.sign(x[1] - x[2])*abs(x[1]**2- x[2]**2)**n - \
        C3*np.sign(x[2] - x[0])*abs(x[2]**2- x[0]**2)**n - \
        C2*np.sign(x[2] - 200)*abs(x[2]**2- 200**2)**n
    return a

def Jacobian(x,C1,C2,C3,C4,C5,n = 0.5):

    A = np.zeros((3,3))
    A[0,0] = np.sign(x[2] - x[0])*np.sign(x[0] - x[2])*C3*abs(x[0])/abs(x[2]**2- x[0]**2)**n -\
        np.sign(x[0] - 200)*np.sign(x[0] - 200)*C1*abs(x[0])/abs(x[0]**2- 200**2)**n
    A[0,2] = np.sign(x[2] - x[0])*np.sign(x[2] - x[0])*C3*abs(x[2])/abs(x[2]**2- x[0]**2)**n
    

    A[1,1] = np.sign(200 - x[1])*np.sign(x[1] - 200)*C4*abs(x[1])/abs(200**2- x[1]**2)**n -\
        np.sign(x[1] - x[2])*np.sign(x[1] - x[2])*C5*abs(x[1])/abs(x[1]**2- x[2]**2)**n
    A[1,2] = -np.sign(x[1] - x[2])*np.sign(x[2] - x[1])*C5*abs(x[2])/abs(x[1]**2- x[2]**2)**n

    
    A[2,0] = -np.sign(x[2] - x[0])*np.sign(x[0] - x[2])*C3*abs(x[0])/abs(x[2]**2- x[0]**2)**n
    
    A[2,1] = np.sign(x[1] - x[2])*np.sign(x[1] - x[2])*C5*abs(x[1])/abs(x[1]**2- x[2]**2)**n
    
    A[2,2] = np.sign(x[1] - x[2])*np.sign(x[2] - x[1])*C5*abs(x[2])/abs(x[2]**2- x[1]**2)**n -\
        np.sign(x[2] - x[0])*np.sign(x[2]-x[0])*C3*abs(x[2])/abs(x[2]**2- x[0]**2)**n - \
        np.sign(x[2] - 200)*np.sign(x[2] - 200)*C2*abs(x[2])/abs(x[2]**2- 200**2)**n
    return A



while True:
    J = Jacobian(x,C1,C2,C3,C4,C5)
    res = residual(x,C1,C2,C3,C4,C5)
    delta = np.linalg.inv(J).dot(res)
    x = x - delta
    if np.linalg.norm(delta, np.inf) < 0.01:
        break
print(x)

def f(x):
    return 0.5*(residual(x,C1,C2,C3,C4,C5)**2).sum()

def J(x):
    return Jacobian(x,C1,C2,C3,C4,C5)

def g(x):
    return J(x).dot(residual(x,C1,C2,C3,C4,C5))

def H(x):
    return Jacobian(x,C1,C2,C3,C4,C5).T.dot(Jacobian(x,C1,C2,C3,C4,C5))


rmax=100
r=1
eta=1./16
tol=1e-5

sol2 = trust_region_dog_leg(f, g, H, x, r, rmax, eta = eta, gtol=tol)
print(sol2)


