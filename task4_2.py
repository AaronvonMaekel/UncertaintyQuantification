import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import scipy as sp
    
def create_stuff(N):
    #dense
    #mat = (N+1) *( np.diagflat(2* np.ones((N)), k=0) - np.diagflat(np.ones((N-1)), k=-1) - np.diagflat(np.ones((N-1)), k=1)  )
    h = 1/(N-1)
    vec = np.ones((N))/(N+1)
    diagonals = [(N+1) *-np.ones((N-1)),(N+1) *2* np.ones((N)),(N+1) *-np.ones((N-1))]
    print(diagonals[1])
    for i in range(1,N-1):
        diagonals[1][i] =  (kappa(h*(i-1)) + kappa(h*(i+1)))/h  #more precise 0.5 * (kappa(h*(i-1))+ 2*kappa(h*(i)) + kappa(h*(i+1)))/h
    print(diagonals[1])
    diagonals[0][0]=0
    diagonals[2][0]=0
    diagonals[0][-1]=0
    diagonals[2][-1]=0
    
    mat =  sp.sparse.diags(diagonals, [-1, 0, 1])
    
    
    # boundary terms
    
    vec[0]=0
    vec[-1]=0

    return mat,vec

def real_sol(x):
    return -0.5 * x**2 + 0.5* x


def kappa(x):
    return 1
    #return 1/(2+x)
def f(x):
    return 1

if __name__ == "__main__":
    N=11
    Nmesh = np.linspace(0,1,N)
    mat,f = create_stuff(N)
    #print(mat)
    #print(f)

    u = sp.sparse.linalg.spsolve(mat,f)
    real_u = real_sol(Nmesh)
    
    plt.plot(Nmesh,u)
    plt.plot(Nmesh,real_sol(Nmesh),color="black")

    plt.show()
    #levels = np.logspace(2,5,10,dtype=np.integer)
    #print(levels)
    #err = []
    #for n in levels:
    #    Nmesh = np.linspace(0,1,n)
     #   mat,f = create_stuff(n)
    #    u = sp.sparse.linalg.spsolve(mat,f)
    #    real_u = real_sol(Nmesh)
    #    err.append(np.sum(np.abs(u- real_u)**2))
    
    
    #plt.plot(levels,err,color="black")
    #plt.yscale("log")
    #plt.show()


    