import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from scipy.fft import dstn



if __name__ == "__main__":
    plt.figure()
    N=1000
    Grid1D = np.linspace(0,1,N,endpoint=False)
    k=N
    #print(np.arange(k))
    lambdas = 1/(0.5+np.arange(k)*np.pi)**2
    
    noise = np.random.normal(size=k)
    #print(lambdas*noise)
    vector_c = dstn(lambdas*noise)

    plt.plot(Grid1D, vector_c, label='Brownian Motion', color='r')  # Plot the second curve in red

    # Add title and labels
    plt.title('Brownian Motion')
    plt.xlabel('Grid')
    plt.ylabel('Value')

    # Show a legend to distinguish the curves
    plt.legend()

    # Show the plot
    plt.show()