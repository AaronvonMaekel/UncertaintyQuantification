import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import seaborn as sns
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
    trials =1000
    Grid1D = np.linspace(0,1,N,endpoint=False)
 
    #i)
    vector_c = matern(Grid1D ,0.5,rho=1)
    real_cov_mat = np.diag(np.ones(N)*vector_c[0],0)
    for i in range(1,len(vector_c)):
        real_cov_mat += np.diag(np.ones(N-i)*vector_c[i],i)
        real_cov_mat += np.diag(np.ones(N-i)*vector_c[i],-i)
    #print(real_cov_mat)
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
    samples= np.empty((trials*2,N))
    heatmap=np.zeros((N,N))



    #ii)
    circle_c = np.concatenate([vector_c, np.flip(vector_c[1:-1])] )
 
    lam   = np.sqrt(N * ifft(circle_c))

    #iii)
    for i in range(trials):
        theta1= random.normal(size=(2*(N-1)))
        theta2= random.normal(size=(2*(N-1)))
        z_complex = fft(lam*(theta1 + theta2 * 1j))/np.sqrt(2*(N-1))
    
        samples[i*2]    = z_complex.real[:N]
        samples[i*2 +1] = z_complex.imag[:N]

    sample_mean = np.mean(samples,axis=0)
    for i in range(trials*2):
        heatmap += np.outer((samples[i]-sample_mean),(samples[i]-sample_mean))
        
    realisation1 = samples[0]
    realisation2 = samples[1]
     
    heatmap/= (2*trials) - 1
    heatmap -= real_cov_mat
   
    plt.figure()
    sns.heatmap(heatmap, cmap='coolwarm')

    
    plt.show()
    
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