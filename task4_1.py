import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


def MC(N,var=False):
    Qs = []
    if var==True:
        Vs = []
    for i in range(len(N)):        
        x = np.random.uniform(size=N[i])/np.random.exponential(size=N[i])
        s= np.mean(x) 
        
        Qs.append(s)
        if var:
            v =np.var(x,ddof=1)
            Vs.append(v)
    if var:
        return np.array(Qs), np.array(Vs)
    else:
        return np.array(Qs)
    


if __name__ == "__main__":
    
    L= 7
    N = 10*np.logspace(0,L-1,L,base=10,dtype=int)
    print(N)
    Qs, Vs = MC(N,True) 
    print(Qs)
    print(Vs)
        # Plotting the curves
    plt.figure(figsize=(10, 6))
    plt.plot(N,Qs,  color='red')
    konf = 2 *np.sqrt(Vs)
    #plt.fill_between(N,  Qs -konf,Qs + konf, color='blue', alpha=0.2)

    plt.xscale("log")
    
    plt.xlabel('Level')
    plt.ylabel('Y_est')
   
    plt.grid()
    # Show plot
    plt.show()
    