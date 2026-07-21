import numpy as np
from models import BSmodel 
from payoff import european_call
from payoff import barrier_call

def beskos_roberts(m,K,T,M,payoff_type = "EC",L = None,seed = None) : 
    """ Computes E[f(X)] by changing the probability
    
    Parameters
    ----------
    m : model
        The model of the EDS (BS, ...)
    K : float
        The strike for Europan options
    T : float
        The maturity of the option of payoff f
    M : int
        The total number of independant paths simulated
    payoff_type : string
        The type of payoff : "EC" (European call), "BC" (Barrier
        call)
    L : float 
        Barrier value
    seed : float
        The seed for randomness
    Returns
    -------
    float
    E[f(X)]
    """

    if(payoff_type not in ["EC","BC"]):
        print("Error : wrong payoff type selected. Please chose between \"BC\" and \"EC\".")
        return None

    if(seed is not None): 
        np.random.seed(seed)

    res = np.zeros(M)
    # Get Y_0 and Y_T under probability Q
    G = np.random.standard_normal(size = M)
    Y_0 = m.g(m.X0)
    Y_T = np.sqrt(T)*G+Y_0

    # Compute the European call payoff if needed
    if payoff_type == "EC" :
        payoff = european_call(m.g_inv(Y_T),K)

    # Compute the exponential
    ex = np.exp(m.h(Y_T)-m.h(Y_0))

    # Compute the last term using Beskos-Roberts
    indicator = np.zeros(M)
    nb_points = np.random.poisson(lam = T*m.c_bound(), size = M) # Drawing the number of points : Poisson law
    
    """ If the payoff is a barrier call 
    or if intermediate values are needed : """
    if payoff_type == "BC":
        payoff = np.zeros(M)
    for j in range(0,M) : 
        x = np.random.uniform(low = 0,high = T,size=nb_points[j])
        x = np.concatenate(([0], x, [T]))
        
        # Brownian bridge :
        index = np.argsort(x)
        x = x[index] #we order the tau_i values
        Y_tau = np.zeros(nb_points[j]+2)
        Y_tau[0] = Y_0
        Y_tau[nb_points[j]+1] = Y_T[j]
        for i in range(1,nb_points[j]+1):
            Y_tau[i] = np.random.normal((Y_T[j]-Y_tau[i-1])*(x[i]-x[i-1])/(T-x[i-1]) + Y_tau[i-1], np.sqrt((x[i]-x[i-1])*(T-x[i])/(T-x[i-1])))


        y = np.random.uniform(low = 0,high = m.c_bound(),size = nb_points[j]) #array of V_i
        
        #Y_taus = np.random.normal((Y_T[j] - Y_0) * x / T + Y_0,np.sqrt(x * (T - x) / T),nb_points[j])
        indicator[j] = np.all(-m.c(Y_tau[1:-1])<=y) #Check that all U_tau are over the -c curve : P(Z=0)=exp(Lambda)

        if payoff_type == "BC" :    
            #x = np.concatenate(([0], x, [T]))
            #Y_taus = np.concatenate(([Y_0], Y_taus, [Y_T[j]]))
            # index = np.argsort(x)
            # x = x[index]
            #Y_tau = Y_tau[index]
            payoff[j] = barrier_call(m,Y_tau, x, m.g(L), K)

        #print(x,Y_tau)
    res = payoff*ex*indicator
    
    return np.mean(res),np.std(res)/np.sqrt(M)

# M = 10**4
#m = BSmodel(100,0.05,0.02)
# def f(x) : 
#     return np.maximum(x-100,0)
# T = 1
# for k in range(4) : 
#     e = beskos_roberts(f,m,T,M,42)   
#     print(e)
    




#beskos_roberts(m,100,1,100,"BC",80,42)
