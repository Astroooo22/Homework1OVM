import numpy as np

def Y_normal():
    y = np.random.randn()
    return y 

def square_returns(T,L,mu,std): #computing square returns in range [0,T]
    r_lst = []
    t_lst = []
    dt = T/L
    t = dt
    r_lst.append(0)
    t_lst.append(0)
    while t <= T:
        r = mu*dt+std*np.sqrt(dt)*Y_normal()
        r_lst.append(r)
        t += dt
    for i in range(len(r_lst)):
        r_lst[i]=(r_lst[i])**2
    return r_lst

def square_returns_mean_var(T,L,mu,std): #computing mean and variance of square returns
    r_lst = square_returns(T,L,mu,std)
    Mean = np.mean(r_lst)
    Var = (np.std(r_lst))**2
    return Mean, Var

def compare(T,L,mu,std): #comparing to results in (7.4) and (7.5)
    Mean, Var = square_returns_mean_var(T,L,mu,std)
    diff_mean = abs(Mean-(std**2*T/L))
    diff_var = abs(Var-(2*std**4*(T/L)**2))
    return diff_mean, diff_var #must both be close 0 to prove approximations (7.4) and (7.5)

print(compare(0.5,1000,0.05,0.3)) #higher L results in more insignificant higher order terms and thus even smaller diff_mean and diff_var

