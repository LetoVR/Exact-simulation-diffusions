import numpy as np
from models import BSmodel

def european_call(X_T, K):  
    """
    Returns the payoff of the European call.

    Parameters
    ----------
    X_T : flaot
        The value of the stock at maturity
    K : float
        The strike

    Returns
    -------
    float
    (S_T-K)_+
    """
    return np.maximum(X_T-K,0)

def barrier_call(m,sample,tau,g_L, K):
    """
    Gives the barrier call payoff with no discretization bias.
    
    Parameters
    ----------
    m : model
        The model of the SDE
    sample : float list
            The P values of the stock sampled on the Beskos-Roberts
            points. sample[0] = Y_0 and sample[P+1] = Y_T.
    tau : float list
            The P tau values (incr order) in which the samples 
            have been made. 
    g_L : float
       The treshold we want the minimum to be greater than,
       evauated by g (with Y = g(X))
    K : float
        The strike for the European call part.
    
    Returns
    -------
    float
    (S_T-K)_+ * P(min Y >= L)
    """

    # Computing the probabilty of the path remaining above the treshold
    p = 1
    for i in range(1,len(sample)):

        # If one of the Y values falls below g(L), we're done
        if sample[i-1] <= g_L or sample[i] <= g_L :
            return 0  
          
        p = p*(1-np.exp(-2/(tau[i]-tau[i-1])*(g_L-sample[i-1])*(g_L-sample[i])))
    
    return european_call(m.g_inv(sample[-1]),K)*p