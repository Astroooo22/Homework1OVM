import numpy as np
import matplotlib.pyplot as plt

#discrete model
def Z_normal():
    z = np.random.randn()
    return z


def discrete_time_model(T,L,S0,std,mu):
    S_lst = []
    t_lst = []
    dt = T/L
    t = dt
    S = S0
    t_lst.append(0)
    S_lst.append(S)
    
    while t <= T:
        S = S*(1+mu*dt+std*np.sqrt(dt)*Z_normal())
        S_lst.append(S)
        t_lst.append(t)
        t += dt
    return S_lst,t_lst

def S_plot_discrete(T,L,S0,std,mu):
    S,t = discrete_time_model(T,L,S0,std,mu)
    plt.figure()
    plt.plot(t,S)
    plt.title("Stock Price Evolution (discrete model)")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.savefig("price_evolution_discrete.png")

def S_sampling_discrete(T,L,S0,std,mu,n_samples):
    i = 1
    S_samples = []
    while i <= n_samples:
        S = discrete_time_model(T,L,S0,std,mu)[0]
        S = S[-1]
        S_samples.append(S)
        i += 1
    samplemean = np.mean(S_samples)
    samplevar = np.var(S_samples)
    return S_samples, samplemean, samplevar

def discrete_hist(T,L,S0,std,mu,n_samples,n_bins):
    S_samples,samplemean,samplevar = S_sampling_discrete(T,L,S0,std,mu,n_samples)
    x_norm = np.linspace(np.min(S_samples),np.max(S_samples),100)
    y_norm = 1/(np.std(S_samples)*np.sqrt(2*np.pi))*np.exp(-0.5*((x_norm-np.average(S_samples))/np.std(S_samples))**2)
    plt.figure()
    plt.hist(S_samples,bins=n_bins,edgecolor="black",density=True,label="Samples")
    plt.plot(x_norm,y_norm,'red',label="Normal distribution")
    plt.legend(fontsize='8')
    plt.title("S_T discrete time model samples histogram")
    plt.xlabel("S_T")
    plt.ylabel("Density (%)")
    plt.savefig("Histogram_discrete.png")
    print(samplemean,samplevar)
    
#continuous model
def continuous_time_model(T,L,S0,std,mu):
    S_lst = []
    t_lst = []
    dt = T/L
    t = dt
    S = S0
    t_lst.append(0)
    S_lst.append(S)
    
    while t <= T:
        S = S0*np.exp((mu-0.5*std**2)*t+std*np.sqrt(t)*Z_normal())
        S_lst.append(S)
        t_lst.append(t)
        t += dt
    return S_lst,t_lst

def S_plot_continuous(T,L,S0,std,mu):
    S,t = continuous_time_model(T,L,S0,std,mu)
    plt.figure()
    plt.plot(t,S)
    plt.title("Stock Price Evolution (continuous model)")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.savefig("price_evolution_continuous.png")

def S_sampling_continuous(T,L,S0,std,mu,n_samples):
    i = 1
    S_samples = []
    while i <= n_samples:
        S = continuous_time_model(T,L,S0,std,mu)[0]
        S = S[-1]
        S_samples.append(S)
        i += 1
    samplemean = np.mean(S_samples)
    samplevar = np.var(S_samples)
    return S_samples, samplemean, samplevar

def continuous_hist(T,L,S0,std,mu,n_samples,n_bins):
    S_samples,samplemean,samplevar = S_sampling_continuous(T,L,S0,std,mu,n_samples)
    x_norm = np.linspace(np.min(S_samples),np.max(S_samples),100)
    y_norm = 1/(np.std(S_samples)*np.sqrt(2*np.pi))*np.exp(-0.5*((x_norm-np.average(S_samples))/np.std(S_samples))**2)
    plt.figure()
    plt.hist(S_samples,bins=n_bins,edgecolor="black",density=True,label="Samples")
    plt.plot(x_norm,y_norm,'red',label="Normal distribution")
    plt.legend(fontsize='8')
    plt.title("S_T continuous time model samples histogram")
    plt.xlabel("S_T")
    plt.ylabel("Density (%)")
    plt.savefig("Histogram_continuous.png")
    print(samplemean,samplevar)

def qqplot(T,L,S0,std,mu,n_samples):
    S_discrete = S_sampling_discrete(T,L,S0,std,mu,n_samples)[0]
    S_continuous = S_sampling_continuous(T,L,S0,std,mu,n_samples)[0]
    q_discrete = np.percentile(S_discrete, np.linspace(0,100,len(S_discrete)))
    q_continuous = np.percentile(S_continuous, np.linspace(0,100,len(S_continuous)))
    plt.figure()
    m = np.std(q_continuous)/np.std(q_discrete)
    c = np.mean(q_continuous) - m * np.mean(q_discrete)
    plt.scatter(q_discrete,q_continuous)
    plt.plot(q_discrete,(m*q_discrete+c),color="red")
    plt.title("Quantile-Quantile Plot")
    plt.xlabel("Discrete Quantiles")
    plt.ylabel("Continuous Quantiles")
    plt.savefig("qq_plot_samples.png")

S_plot_discrete(1,10,170,0.344,0.190)
discrete_hist(1,10,170,0.344,0.190,500,10)
S_plot_continuous(1,10,170,0.344,0.190)
continuous_hist(1,10,170,0.344,0.190,500,10)
qqplot(1,10,170,0.344,0.190,500)