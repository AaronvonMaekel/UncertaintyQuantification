import numpy as np
import matplotlib.pyplot as plt
import timeit
def f(x,s=1):
    y = np.power((2+ 1/(2*s)),s) * np.prod(np.power(x , 1+ 1/(2*s)))
    return y

def trapezrule(N,s=1):

    h=1/N
    xs = np.linspace(0,1,N+1,endpoint=True)
    area=0 
    N_combs = (N+1)**s
    calcs = np.power((N+1),np.linspace(0,s-1,s))
    for i in range(N_combs):
        #INCREASE S_IND
        s_ind = np.mod(np.floor(i/calcs),N+1).astype(int)
        
        #CALCULATE BORDERS
        borders = (s_ind ==0).sum() +  (s_ind ==1).sum() 

        area += (1/np.power(2,borders)) * f(xs[s_ind],s)
    
    area *= np.power(h,s)
    return area

def plots(s=1):
    Ns = np.logspace(1,3,10)
    y=[]
    for i in Ns: 
        
        y.append(trapezrule(int(i),s)-1)

    plt.figure()
    plt.plot(Ns,y)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

start = timeit.timeit()
plots(2)
print(timeit.timeit() - start)
#trapezrule(10)