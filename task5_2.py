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
    kappa = np.zeros(len(x))
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
    TRAPEZ_POINTS = 21
    for i in range(len(diagonals[1])):
        #diagonals[1][i] = 1/(h**2) * integrate.quad(lambda x: kappa(x, q, omegas), h*(i-1), h*(i+1))[0]
        x = np.linspace(h*(i-1), h*(i+1),TRAPEZ_POINTS)
        y = kappa(x, q,  omegas)
        diagonals[1][i]  =    np.trapz(y, x)
    for i in range(len(diagonals[0])):
        #diagonals[0][i] = -1/(h**2) * integrate.quad(lambda x: kappa(x, q, omegas), h*(i), h*(i+1))[0]
       
        x = np.linspace( h*(i), h*(i+1),TRAPEZ_POINTS)
        y = kappa(x, q, omegas)
        diagonals[0][i]  =    np.trapz(y, x)

        diagonals[2][i] = diagonals[0][i]
    # Impose boundary conditions.
    diagonals[0][0] = 0
    diagonals[2][0] = 0
    diagonals[0][-1] = 0
    diagonals[2][-1] = 0
    
    mat = sp.sparse.diags(diagonals, [-1, 0, 1])

    return mat


def create_rhs(N: int, f):
    h = 1 / (N - 1)
    TRAPEZ_POINTS = 21
    vec = np.empty((N))

    for i in range(len(vec)):
        #vec[i] = integrate.quad(lambda x: f(x)*(1-abs(h*i-x)/h), h*(i-1), h*(i+1))[0]
        x = np.linspace(h*(i-1), h*(i+1),TRAPEZ_POINTS)
        y = f(x)*(1-abs(h*i-x)/h)
        vec[i] =    np.trapz(y, x)
    # Impose boundary conditions.
    vec[0] = 0
    vec[-1] = 0

    return vec



if __name__ == "__main__":
    h=0.01
    
    N = int(1/h) +1 
    Nmesh = np.linspace(0,1,N)
    

    s = 10
    #########c
    #levels = np.logspace(2,4.5,6,dtype=np.integer)
    #f = create_rhs(N, lambda x: 1)
    #Qs = np.array([1.1,2,10])
    #MC_estimate = 0
    # Perform vanilla MC.
    #MC_var  = np.empty((len(Qs),len(levels)))
    #i = 0
    #k = 0
    #for q in Qs:
    #    i = 0
    #    for N_samples in levels:
    #        MC_samples = np.empty(N_samples)
    #        for j in tqdm(range(N_samples)):
    #            omegas = random.standard_normal(s)
    #            mat = create_stiffness_matrix(N, q, omegas)

    #            u = sp.sparse.linalg.spsolve(mat, f)

    #            MC_estimate += u[int((N - 1) * 0.7)]
    #            MC_samples[j] = u[int((N - 1) * 0.7)]
    #        MC_var[k,i] = np.var(MC_samples,ddof=1)    
    #        MC_estimate /= N_samples
    #        print(f"MC estimate: {MC_estimate}")
    #        i += 1
    #    k+= 1

    #
    ###print(f"MC variance: {MC_var}")
    #plt.plot(levels,MC_var[0],color="black",label="q=1.1")
    #plt.plot(levels,MC_var[1],color="blue",label="q=2")
    #plt.plot(levels,MC_var[2],color="red",label="q=10")
    #plt.legend()
    #plt.yscale("log")
    #plt.xscale("log")
    #plt.show()
    #######################d
    q = 2
    hmesh = 0.01*np.logspace(-0,-5,10,base=2)
    MC_mean  = np.empty((len(hmesh))) 
    N_samples = 1000
    print(hmesh)
    i = 0
    MC_samples = np.empty(N_samples)
    for h in hmesh:
        N = int(1/h) +1 
        Nmesh = np.linspace(0,1,N)
        f = create_rhs(N, lambda x: 1)                    
        for j in tqdm(range(N_samples)):
            omegas = random.standard_normal(s)
            mat = create_stiffness_matrix(N, q, omegas)

            u = sp.sparse.linalg.spsolve(mat, f)
            MC_samples[j] = u[int((N - 1) * 0.7)]
        MC_mean[i] = np.mean(MC_samples)
        print(f"MC estimate: {MC_mean}")
        i+=1
    plt.plot(hmesh, abs(MC_mean[:-1] - MC_mean[-1]),color="black",label="error")

    plt.yscale("log")
    plt.xscale("log")
    plt.show()
