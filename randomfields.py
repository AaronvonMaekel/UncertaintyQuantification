import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft
from scipy.special import kv,gamma

def p(x):
    return np.exp(-4*np.abs(x))

def matern(x, nu, rho, var2=1):
    if nu == 0.5:
        return var2 * np.exp(-np.sqrt(2)*x/rho)
    if nu == 'inf':
        return var2 * np.exp(- (x**2)/(rho**2))
    if nu == 1:
        #print(var2*(2*x[1:]/rho) * kv(nu,x[1:]))
        return np.append(1 ,var2*(2*x[1:]/rho) * kv(nu,2*x[1:]/rho))
    else:
        return np.append(1 ,var2/((2**(nu-1))*gamma(nu)) * ((2*np.sqrt(nu)*x[1:]/rho)**nu) * kv(nu,2*np.sqrt(nu)*x[1:]/rho))
if __name__ == "__main__":
    #a)
    N=500
 
    Grid1D = np.linspace(0,1,N,endpoint=False)
 
    #i)
    vector_c = matern(Grid1D ,2,rho=0.1)
    #vector_c = p(Grid1D -Grid1D[0]* np.ones_like(Grid1D))
    plt.figure()
     
    plt.plot(Grid1D, vector_c, label='Cov Function', color='r')  # Plot the second curve in red

    # Add title and labels
    plt.title('Covariance Function')
    plt.xlabel('Grid')
    plt.ylabel('Value')

    # Show a legend to distinguish the curves
    plt.legend()

    # Show the plot
    plt.show()

    #ii)
    circle_c = np.concatenate([vector_c, np.flip(vector_c[1:-1])] )
 
    lam   = np.sqrt(N * ifft(circle_c))
    
    #iii)
    theta1= random.normal(size=(2*(N-1)))
    theta2= random.normal(size=(2*(N-1)))
    z_complex = fft(lam*(theta1 + theta2 * 1j))/np.sqrt(2*(N-1))
 
 

    realisation1 = z_complex.real[:N]
    realisation2 = z_complex.imag[:N]
    
    plt.figure()
    plt.plot(Grid1D, realisation1, label='First realisation', color='b')  # Plot the first curve in blue
    plt.plot(Grid1D, realisation2, label='Second Realisation', color='r')  # Plot the second curve in red

    # Add title and labels
    plt.title('2 Realisations over a 1D Grid')
    plt.xlabel('Grid')
    plt.ylabel('Value')

    # Show a legend to distinguish the curves
    plt.legend()

    # Show the plot
    plt.show()