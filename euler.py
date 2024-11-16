import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
u_zero = [0.5,2]
T = 6
delta = 0.2
hmesh = [0.6,0.06,0.006,0.0006] # lieber duplzieren statts mal 10
Nmesh = np.logspace(2,4,2,dtype=int)
results = np.empty((len(hmesh),len(Nmesh)))
results_var = np.empty((len(hmesh),len(Nmesh)))
def f(u):
    return np.array([ u[0] - u[0]*u[1], u[0]*u[1] - u[1]])

di = 0
for h in hmesh:
    print("H",h)
    steps = int(T/h)
    dj = 0
    for N in Nmesh:
        Q=[]
        print("N",N)
        for i in range(N):
            u_noise = u_zero + random.uniform(low=-delta, high=delta, size=2)

            for t in range(steps):
                u_noise = u_noise + h*f(u_noise)
            Q.append(u_noise[0])  
        Q_est = np.mean(Q) 
        #print(h,N,Q_est)
        results[di,dj] = Q_est
        results_var[di,dj] = 2* np.sqrt(np.var(Q)) #95%
        
        dj+=1
    di+=1
print(results)
print(results_var)
### real Q:
#real_H = 0.000006
#steps = int(T/real_H)#

#u_noise = u_zero
#for t in range(steps):
#    u_noise = u_noise + real_H*f(u_noise)
          
#    Q_real = u_noise[0]
#print("real Q",Q_real)      
slide_Q = 1.3942


# Plotting the curves
plt.figure(figsize=(10, 6))

plt.plot(hmesh, results[:,0], label='small N', color='red')
plt.plot(hmesh, results[:,1], label='Big N', color='blue')
plt.xscale('log')
# Adding confidence intervals
#plt.fill_between(hmesh, results[:,0] -results_var[:,0], results[:,0]  + results_var[:,0], color='red', alpha=0.2)
plt.fill_between(hmesh,  results[:,1] -results_var[:,1], results[:,1]  + results_var[:,1], color='blue', alpha=0.2)
plt.gca().invert_xaxis()
# Adding labels and title
plt.title('low and big N  with Confidence Intervals')
plt.xlabel('h')
plt.ylabel('Q')
plt.legend()
plt.grid()
plt.hlines(slide_Q,hmesh[0],hmesh[-1],colors="black")
# Show plot
plt.show()