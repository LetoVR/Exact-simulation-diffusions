from monte_carlo import Monte_Carlo
import numpy as np
import matplotlib.pyplot as plt

from black_scholes import bs_expectation
from models import BSmodel
from beskos_roberts import beskos_roberts
from payoff import european_call
from payoff import barrier_call



def validate_Euler_EC(m,K,T,N,M,ref,seed = None) : 
    """ Validates or not the Euler scheme for a 
    European call price estimated by Monte-Carlo 
    compared to the true one.
    
    Parameters
    ----------
    m : model
        The model under which the EDS is
    f : float -> float
        The payoff function
    N : int
        The number of discretization steps
    M : int 
        The number of simulated paths in the Monte-Carlo
    ref : float
        The true option price under B-S
    Returns
    -------
    bool
    true if the Euler price is close enough to the true price (i.e. gap < 3*vol)
    """
    e,vol = Monte_Carlo(m,K,T,N,M,"EC",seed)
    gap = abs(ref-e)
    n_sigma = gap / vol

    if n_sigma>=4 :
        print(f"MC: {e:.4f} ± {vol:.4f}, ref: {ref:.4f}, gap = {n_sigma:.2f}σ")
    return n_sigma<4

def first_test_Euler(seed = None) : 
    S0 = 100
    T = 1
    # Initialization various values of the parameters into arrays
    N_tab = np.array([10,100,1000])
    M_tab = np.array([10,100,1000,10000,100000])
    sigma_tab = np.array([0.1, 0.2, 0.3, 0.5])
    r_tab = np.array([0.01,0.05,0.1,0.4])
    K_tab = np.array([S0,S0-20,S0+20])
    
    print("Running initial coarse MC test for Euler scheme with a wide range of parameters.")
    print("Pinting all failures and their parameters.")
    for r in r_tab : 
        for sigma in sigma_tab : 
            for K in K_tab : 
                m = BSmodel(S0,r,sigma)
                ref = bs_expectation(S0,K,r,sigma,T)  #The true price
        
                #print(f"For K = {K}, r = {r}, sigma = {sigma} : ")

                for N in N_tab : 
                    for M in M_tab : 
                        #print(f" - {N} time steps, {M} paths simulated :")
                        if not validate_Euler_EC(m,K,T,N,M,ref,seed) : 
                            print(f"For K = {K}, r = {r}, sigma = {sigma} : ")
                            print(f" - {N} time steps, {M} paths simulated :")
                            print("Failure")
                        

#first_test_Euler(42)

                        
def final_test_Euler(seed = None) : 
    S0 = 100
    T = 1
    # Initialization various values of the parameters into arrays
    N_tab = np.array([100,1000])
    M_tab = np.array([100,1000,10000,100000])
    sigma_tab = np.array([0.1, 0.2, 0.3, 0.5])
    r_tab = np.array([0.01,0.05,0.1])
    K_tab = np.array([S0,S0-20,S0+20])


    print("Running final MC test for Euler scheme: printing all failures with their parameters")
    for r in r_tab : 
        for sigma in sigma_tab : 
            for K in K_tab : 
                m = BSmodel(S0,r,sigma)
                ref = bs_expectation(S0,K,r,sigma,T)  #The true price
                def f(x) : 
                    return european_call(x,K)
                #print(f"For K = {K}, r = {r}, sigma = {sigma} : ")

                for N in N_tab : 
                    for M in M_tab : 
                        #print(f" - {N} time steps, {M} paths simulated :")
                        if not validate_Euler_EC(m,K,T,N,M,ref) : 
                            print(f"For K = {K}, r = {r}, sigma = {sigma} : ")
                            print(f" - {N} time steps, {M} paths simulated :")
                            print("Failure")


#final_test_Euler(42)

def validate_BR_EC(m,K,T,M,ref,payoff_type,L,seed):
    """ Validates or not the Beskos-Roberts price of a European
    call estimated by Monte-Carlo compared to the true one.
    
    Parameters
    ----------
    m : model
        The model under which the EDS is
    f : float -> float
        The payoff function
    T : float
        Maturity
    M : int 
        The number of simulated paths in the Monte-Carlo
    ref : float
        The true option price under B-S
    payoff-type : string
        The type of the payoff we want
    Returns
    -------
    bool
    true if the Beskos-Roberts price is close enough to the true price 
    """
    e,sigma = beskos_roberts(m,K,T,M,payoff_type,L,seed)
    print(f"- for {M} paths simulated :") 
    print(f"M-C price : {e:.4f}, standard-deviation : {sigma:.4f}")
    return (e + sigma > ref) and (e-sigma < ref)
    


def test_Beskos_Roberts_BS_European_Call(seed = None) : 
    S0 = 100
    T = 1
    # Initialization various values of the parameters into arrays
    M_tab = np.array([1000,100000])
    sigma_tab = np.array([0.1, 0.3, 0.5])
    r_tab = np.array([0.01,0.05,0.1])
    K_tab = np.array([S0,S0+20])

    print("Running MC test for Beskos-Roberts method on Black-Scholes model for a Europan call")
    print("Printing results (P: passed/F: failed) with their parameters...")

    for r in r_tab : 
        for vol in sigma_tab :  
            for K in K_tab : 


                m = BSmodel(S0,r,vol)
                true_price = bs_expectation(S0,K,r,vol,T)
                print(f"For a drift of {r}, a volatility of {vol} and a strike {K} : ")
                print(f"The true price is {true_price:.4f},")
                for M in M_tab :
                    if validate_BR_EC(m,K,T,M,true_price,"EC",None,seed):
                        print("Result : P")
                    else : 
                        print("Result : F")

#test_Beskos_Roberts_BS_European_Call(42)

def initial_validate_BR_BC(m,K,T,M,ref,payoff_type,L,seed):
    """ Validates or not the Beskos-Roberts price of a European
    call estimated by Monte-Carlo compared to the true one.
    
    Parameters
    ----------
    m : model
        The model under which the EDS is
    f : float -> float
        The payoff function
    T : float
        Maturity
    M : int 
        The number of simulated paths in the Monte-Carlo
    ref : float
        The true option price under B-S
    payoff-type : string
        The type of the payoff we want
    Returns
    -------
    bool
    true if the Beskos-Roberts price is close enough to the true price 
    """
    e,sigma = beskos_roberts(m,K,T,M,payoff_type,L,seed)
    print(f"- for {M} paths simulated :") 
    print(f"M-C price : {e:.4f}, standard-deviation : {sigma:.4f}")
    return e<ref

""" The initital tests below allow us to check if, when we 
simulate a lot of paths (1 million), the price of the barrier
call is always under the price of the european call.  The only 
thing we look at is : 
True price >= M-C price, because it should be always the case
due to the barrier that brings some paths to a price of 0. To 
ensure the relevance of the test, we tried with a very little
L value that makes the Barrier call and the European call close."""
def initial_test_Beskos_Roberts_BS_Barrier_Call(seed = None) : 
    S0 = 100
    T = 1
    # Initialization various values of the parameters into arrays
    M_tab = np.array([1000000])
    sigma_tab = np.array([0.1, 0.3, 0.5])
    r_tab = np.array([0.01,0.05,0.1])
    K_tab = np.array([S0,S0+20])

    """For the inital test, we take a very small value of L 
    to assert the barrier call converges to the european call"""
    L = S0/100


    print("Running MC initial test for Beskos-Roberts method on Black-Scholes model for a barrier call")
    print("For the inital test, we take a small L = S0/100.")
    print("Printing results (P: passed/F: failed) with their parameters...")

    for r in r_tab : 
        for vol in sigma_tab :  
            for K in K_tab : 


                m = BSmodel(S0,r,vol)
                true_price = bs_expectation(S0,K,r,vol,T)
                print(f"For a drift of {r}, a volatility of {vol} and a strike {K} : ")
                print(f"The true price is {true_price:.4f},")
                for M in M_tab :
                    if initial_validate_BR_BC(m,K,T,M,true_price,"BC",L,seed):
                        print("Result : P")
                    else : 
                        print("Result : F")

initial_test_Beskos_Roberts_BS_Barrier_Call(42)


""" This second set of tests allows us to check that the 
M-C price for a Barrier option decreases when L grows and
that it converges to the european call true price when L->0."""
def second_test_Beskos_Roberts_BS_Barrier_Call(seed = None) : 
    # We chose a fixed configuration of parameters : 
    S0 = 100
    T = 1
    sigma = 0.3
    r = 0.05
    m = BSmodel(S0,r,sigma)

    M = 10000
    # We draw 3 different curb depending on K (OM/AM/IM)
    K_tab = np.array([S0-20,S0,S0+20])
    

    # Computing the 3 different true prices of european call
    ref_tab = bs_expectation(S0,K_tab,r,sigma,T)
    
    # We allow L to vary in ]0,S0]
    # 40 points near S0/1000, 20 in middle, 40 near S0
    L_tab = np.concatenate([
        np.logspace(np.log10(S0/1000), np.log10(S0/5), 5),
        np.logspace(np.log10(S0/5), np.log10(S0/1.7), 5),
        np.logspace(np.log10(S0/1.7), np.log10(S0), 20)
    ])
    L_tab = np.unique(L_tab)  # remove any duplicates

    plt.figure(figsize=(10, 6))
    for k in range(len(K_tab)) : 
        e_tab,sigma_tab = np.zeros(len(L_tab)),np.zeros(len(L_tab))
        for i in range(len(L_tab)) : 
            e_tab[i], sigma_tab[i] = beskos_roberts(m,K_tab[k],T,M,"BC",L_tab[i],seed)

        plt.plot(L_tab, e_tab, label = f'K = {K_tab[k]}', marker = 'o', markersize=3)
        plt.axhline(y = ref_tab[k], color = 'C'+str(k), linestyle ='--', alpha = 0.5, label = f'True price for K={K_tab[k]}')

    
    plt.xlabel('L (Barrier level)')
    plt.ylabel('Monte-Carlo Price')
    plt.title('Barrier Call Price vs European call true price')
    plt.legend()

    params_text = f'Parameters : \n$S_0$ = {S0}\n$T$ = {T}\n$\\sigma$ = {sigma}\n$r$ = {r}\n$M$ = {M}'
    plt.text(0.98, 0.97, params_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend(loc='lower left')

    plt.grid(True, alpha=0.3)
    plt.savefig('BR_BC_convergence_and_monotony.pdf')
    #plt.show()




second_test_Beskos_Roberts_BS_Barrier_Call(42)

""" This final test asserts that our computation of 
the barrier call price, thanks to the q function, 
is correct. We compare it to the Euler scheme that
discretizes the path to estimate its payoff."""
def final_test_Beskos_Roberts_BS_Barrier_Call(seed = None) : 
    # We chose a fixed configuration of parameters : 
    S0 = 100
    T = 1
    sigma = 0.3
    r = 0.05
    m = BSmodel(S0,r,sigma)

    M = 100000
    K = S0
    L = S0*95/100

    N_tab = np.array([50,100,200,500,1500,10000])
    
    # We compute the barrier price under discretization method : 
    biased_estimator_mean, biased_estimator_std = np.zeros(len(N_tab)),np.zeros(len(N_tab))
    for i,N in enumerate(N_tab) :   
        # We take different seed for each call to ensure independance  
        biased_estimator_mean[i], biased_estimator_std[i] = Monte_Carlo(m,K,T,N,M,"BC",L,None)

    # We compute the barrier price under Beskos-Roberts
    unbiased_estimator_mean, unbiased_estimator_std = beskos_roberts(m,K,T,M,"BC",L,seed)

    # Now we plot both against 1/sqrt(N), because that's the expected convergence
    # rate of Euler. We should have 2 straight lines, one flat, the other intersecting
    # the first as N --> infinity.
    plt.figure(figsize=(10, 6))
    sqrt_N_tab = 1/np.sqrt(N_tab)
    plt.plot(sqrt_N_tab, biased_estimator_mean, label='Euler', marker='o', markersize=6)
    plt.fill_between(sqrt_N_tab, 
                     biased_estimator_mean - biased_estimator_std,
                     biased_estimator_mean + biased_estimator_std,
                     alpha=0.2)
    
    plt.axhline(y=unbiased_estimator_mean, color='C1', linestyle='--', label='Beskos-Roberts')
    plt.fill_between(sqrt_N_tab,
                     unbiased_estimator_mean - unbiased_estimator_std,
                     unbiased_estimator_mean + unbiased_estimator_std,
                     color='C1', alpha=0.2)

    coeffs = np.polyfit(sqrt_N_tab, biased_estimator_mean, 1)
    regression_line = np.polyval(coeffs, sqrt_N_tab)
    plt.plot(sqrt_N_tab, regression_line, color='C0', linestyle=':', linewidth=2, 
             label=f'Linear fit')


    plt.xlabel('1/sqrt(N) (Number of discretization steps)')
    plt.ylabel('Barrier Call Price')
    plt.title('Convergence: Euler Discretization vs Beskos-Roberts')
    plt.legend()
    plt.grid(True, alpha=0.3)

    params_text = f'Regression\'s intercept: {coeffs[1]:.4f}\nBeskos-Roberts: {unbiased_estimator_mean:.4f}\nBeskos-Roberts std: {unbiased_estimator_std:.4f}'
    plt.text(0.5, 0.975, params_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig('Euler_vs_BR_BC_convergence.pdf')
    #plt.show()

final_test_Beskos_Roberts_BS_Barrier_Call(42)