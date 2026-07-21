import numpy as np
from models import BSmodel
from black_scholes import bs_expectation
import matplotlib.pyplot as plt
from payoff import european_call
from payoff import barrier_call


def Monte_Carlo(model,K, T, N, M,payoff_type = "EC",L = None, seed = None):
    """ Simulates paths solution of a given EDS by discretizing it
    using a Euler scheme. 
    
    Parameters
    ----------
    model : model
            The model of the EDS (BS, ...)
    K : float
        The strike
    T : float
        The maturity of the option of payoff f
    N : int 
        The number of steps for discretization
    M : int
        The total number of independant paths simulated
    payoff_type : string
        "EC" for the European call, "BC" for the barrier
        call
    L : float
        The barrier of the barrier call
    Returns
    -------

    if payoff_type = "EC" :
    float, float
    E[f(X)],Var(f(X)) the expectation and variance of the payoff of the solution

    if payoff_type = "BC : 
    float array : M paths discretized with a step T/N
    """

    if (seed is not None) : 
        np.random.seed(seed)

    path = np.empty((N+1,M))
    h = T/N
    
    # Initiating first row to X0
    path[0,:] = model.X0

    for k in range(1,N+1):
        path[k,:] = path[k-1,:] + model.b(path[k-1,:])*h + model.sigma(path[k-1,:])*np.sqrt(h)*np.random.standard_normal(M)
    
    value = european_call(path[N,:],K)
    
    if payoff_type == "EC" : 
        return np.mean(value),np.std(value)/np.sqrt(M)

    if payoff_type == "BC" : 
        min_tab = np.min(path, axis=0) #array of minimums for each path  
        indicator = (min_tab>=L).astype(int) # bool array : 0 if min<L, 1 else
        payoff = value*indicator
        return np.mean(payoff), np.std(payoff)/np.sqrt(M)
        




# # Plot the bias to compare to 1/N 
# M = 10**4
# m = BSmodel(100,0.4,0.01)
# def f(x) : 
#     return np.maximum(x-100,0)
# T = 1
# N = np.array([10,20,50,100,200,500,1000])
# dt = 1.0/N

# e = np.array([Monte_Carlo(m,f,T,N[i],M,42)[0] for i in range(len(N))]) 
# true_mu = bs_expectation(m.X0,100,m.r,m.vol,T)
# bias = np.abs(e-true_mu)
# slope,intercept =   np.polyfit(np.log(dt),np.log(bias),deg = 1)
# print(f"Empirical weak order : {slope : .3f}")


# plt.figure(figsize=(6, 4))
# plt.loglog(dt, bias, 'o', label='measured |bias|')
# plt.loglog(dt, np.exp(intercept) * dt**slope, '-',
#            label=f'fit: slope={slope:.2f}')
# plt.loglog(dt, dt * (bias[0]/dt[0]), '--', alpha=0.5,
#            label='slope 1 reference')
# plt.xlabel(r'$\Delta t = T/N$')
# plt.ylabel(r'$|\hat\mu - \mu^*|$')
# plt.legend()
# plt.title('Euler weak convergence: BS, r=0.4, σ=0.1')
# plt.tight_layout()
# plt.savefig('euler_convergence.pdf')
