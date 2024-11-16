import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

import scipy as sp
import scipy.integrate as integrate

from tqdm import tqdm 

def kappa(
    x: float,  # Evaluation point
    q: float,
    omegas: np.array,
) -> float:
    kappa = 0
    for j in range(1, len(omegas) + 1):
        # Compute corresponding basis function.
        a = (1 / j ** q) * np.sin(2 * j * np.pi * x)
        kappa += omegas[j - 1] * a
    
    return np.exp(kappa)


def create_stiffness_matrix(
    N: int,
    q: float,
    omegas: np.array,
):
    h = 1 / (N - 1)
    diagonals = [np.empty((N - 1)), np.empty((N)), np.empty((N - 1))]
    
    for i in range(len(diagonals[1])):
        diagonals[1][i] = 1/(h**2) * integrate.quad(lambda x: kappa(x, q, omegas), h*(i-1), h*(i+1))[0]
    
    for i in range(len(diagonals[0])):
        diagonals[0][i] = -1/(h**2) * integrate.quad(lambda x: kappa(x, q, omegas), h*(i), h*(i+1))[0]
        diagonals[2][i] = -1/(h**2) * integrate.quad(lambda x: kappa(x, s, omegas), h*(i), h*(i+1))[0]
    
    # Impose boundary conditions.
    diagonals[0][0] = 0
    diagonals[2][0] = 0
    diagonals[0][-1] = 0
    diagonals[2][-1] = 0
    
    mat = sp.sparse.diags(diagonals, [-1, 0, 1])

    return mat


def create_rhs(N: int, f):
    h = 1 / (N - 1)

    vec = np.empty((N))

    for i in range(len(vec)):
        vec[i] = integrate.quad(lambda x: f(x)*(1-abs(h*i-x)/h), h*(i-1), h*(i+1))[0]

    # Impose boundary conditions.
    vec[0] = 0
    vec[-1] = 0

    return vec


if __name__ == "__main__":
    N = 1001
    Nmesh = np.linspace(0,1,N)
    
    N_samples = 10

    s = 10
    q = 2

    f = create_rhs(N, lambda x: 1)
    
    MC_estimate = 0
    # Perform vanilla MC.
    for _ in tqdm(range(N_samples)):
        omegas = random.standard_normal(s)
        mat = create_stiffness_matrix(N, q, omegas)

        u = sp.sparse.linalg.spsolve(mat, f)

        MC_estimate += u[int((N - 1) * 0.7)]
    MC_estimate /= N_samples
    
    print(f"MC estimate: {MC_estimate}")
    