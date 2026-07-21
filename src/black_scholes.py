""" First python file where the method is implemented on the most basic B-S model """
""" The advantage is that we can check if everything works, as we have the exact results for this model """

import numpy as np
from scipy.stats import norm

# The true price of a European Call in B-S model :
def bs_call_price(S0, K, r, sigma, T):
    """
    Black-Scholes price for a European call option.

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike
    r : float
        Risk-free actualization rate
    sigma : float
        Volatility
    T : float
        Maturity of the Call
    Returns
    -------
    float 
        E[(S_T-K)_+] under the risk-neutral probability, the fair price of a European Call 
    """


    d_1 = (np.log(S0/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d_2 = d_1 - sigma*np.sqrt(T)

    return S0*norm.cdf(d_1) - K*np.exp(-r*T)*norm.cdf(d_2)


def bs_expectation(S0, K, r, sigma, T):
    return np.exp(r*T)*bs_call_price(S0, K, r, sigma, T)


if __name__ == "__main__" : 
    price = bs_call_price(S0 = 100, K = 100, r = 0.05, sigma = 0.2, T =  1.0)
    print(f"B-S call fair price (ATM, r=5%, sigma=20%, T=1y): {price : .4f}")



