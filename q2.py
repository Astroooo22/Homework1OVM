import numpy as np
import matplotlib.pyplot as plt
import csv

####2.1####
def X_uniform(low, high): #defining uniform random variable X
    x = np.random.uniform(low,high)
    return x

def Z_uniform(n,low,high): #sum Z_n from slide 12 lecture 3
    z = 0
    i = 1
    while i<=n:
        x = X_uniform(low,high)
        z = z + x
        i += 1
    z = (z-n/2)/np.sqrt(n/12) 
    return z

def S_uniform(m,n,low,high): #drawing m samples from Z
    S = []
    j = 1
    while j <= m:
        S.append(Z_uniform(n,low,high))
        j += 1
    return S
 
def bins_uniform(nbins,m,n,low,high): #creating bins to plot pdf of continuous RV
    s = S_uniform(m,n,low,high)
    min = np.min(s)
    max = np.max(s)
    bin = (max-min)/nbins
    bins = []
    x_vals = []
    for l in range(nbins+1):
        bins.append(min + bin*l)
    for r in range(1,nbins*2,2):
        x_vals.append(min+bin*r/2)
    x_vals = (x_vals-np.mean(x_vals))/(np.std(x_vals))
    return bins,s,x_vals

def uniform_xy_vals(nbins,m,n,low,high): #computing x and y values for pdf plot
    bns,s,x_vals = bins_uniform(nbins,m,n,low,high)
    y_valss = np.zeros(nbins)
    y_vals = []
    for p in range(len(s)):
        q = 0
        while (s[p]-bns[q+1]) > 0 and q < (len(bns)-2):
                    q += 1
        y_valss[q] += 1
    for s in range(len(y_valss)):
        y_vals.append(y_valss[s])
    return x_vals,y_vals
        
def uniform_pdf(nbins,m,n,low,high): #plotting probability density function
    x,y = uniform_xy_vals(nbins,m,n,low,high)
    y = y/np.trapz(y,x)
    plt.plot(x,y)
    plt.ylabel("probability density")
    plt.xlim(-2,2)
    plt.ylim(0,1)
    plt.xlabel("x")
    plt.title("Z_n pdf estimate - "+str(m)+" samples, "+str(nbins)+" bins"+", n = "+str(n))
    plt.savefig("pdf"+str(n)+".png")

uniform_pdf(30,1000000,1,0,1)
uniform_pdf(30,100000,2,0,1)
uniform_pdf(30,100000,3,0,1)
uniform_pdf(30,100000,30,0,1)
uniform_pdf(30,100000,100,0,1)

####2.2####
#dailydata
with open('DailyData - STOCK_US_XNAS_AAPL.csv', 'r') as file: #reading csv file
    csv_reader = csv.reader(file)

    next(csv_reader)

    close_prices = []

    for row in csv_reader:
        if row:
            close_prices.append(row[4])

def price_mvt(n_days): #computing x and y for price movement graph
    x = []
    y = []
    for i in range(len(close_prices)):
        x.append(i+1)
        y.append(float(close_prices[i]))
    return x,y

def plot_price(n_days): #plotting price movement graph
    x,y = price_mvt(n_days)
    plt.figure()
    plt.plot(x,y)
    plt.title("Daily Price Movement")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend(["STOCK_US_XNAS_AAPL"],fontsize="10")
    plt.savefig("price_movement.png")

def log_return(n_days): #re-computing y for logreturn graph
    x,y = price_mvt(n_days)
    x_lr = x[1:]
    y_lr = []
    for j in range(1,len(y)):
        lr = np.log(y[j]/(y[j-1]))
        y_lr.append(lr)
    return x_lr,y_lr

def plot_logreturn(n_days): #plotting logreturn graph
    x,y = log_return(n_days)
    plt.figure()
    plt.plot(x,y)
    plt.title("Log Daily Return")
    plt.xlabel("Day")
    plt.ylabel("Logreturn")
    plt.legend(["STOCK_US_XNAS_AAPL"],fontsize="10")
    plt.savefig("logreturn.png")

def hist_plot(n_bins, n_days): #plotting histogram + normal distribution curve
    lr = log_return(n_days)[1]
    x_norm = np.linspace(np.min(lr),np.max(lr),100)
    y_norm = 1/(np.std(lr)*np.sqrt(2*np.pi))*np.exp(-0.5*((x_norm-np.average(lr))/np.std(lr))**2)
    plt.figure()
    plt.hist(lr,bins=n_bins,edgecolor="black",density=True,label="Samples")
    plt.plot(x_norm,y_norm,'red',label="Normal distribution")
    plt.legend(fontsize='8')
    plt.title("Log Daily Return Histogram")
    plt.xlabel("Returns")
    plt.ylabel("Density (%)")
    plt.savefig("Histogram.png")

def mean_and_variance(n_days):
    lr = log_return(n_days)[1]
    mean = np.mean(lr)
    variance = np.var(lr)
    return mean, variance

def qq_plot(n_days): #plotting quantile-quantile plot
    lr = log_return(n_days)[1]
    q_norm = np.percentile(np.random.normal(0,1,len(lr)), np.linspace(0,100,len(lr)))
    q_sample = np.percentile(lr, np.linspace(0,100,len(lr)))
    #fitting straight line
    m = np.std(q_sample) / np.std(q_norm)
    c = np.mean(q_sample) - m * np.mean(q_norm)
    plt.figure()
    plt.scatter(q_norm, q_sample)
    plt.plot(q_norm,(m*q_norm+c),color="red")
    plt.title("Quantile-Quantile Plot")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.savefig("QQ_Plot.png")


plot_price(len(close_prices))
plot_logreturn(len(close_prices))
hist_plot(10,len(close_prices))
print(mean_and_variance(len(close_prices)))
qq_plot(len(close_prices))


#weekly data
with open('WeeklyData - STOCK_US_XNAS_AAPL.csv', 'r') as file: #reading csv file
    csv_reader = csv.reader(file)

    next(csv_reader)

    close_prices = []

    for row in csv_reader:
        if row:
            close_prices.append(row[4])

def price_mvt_weekly(n_weeks): #computing x and y for price movement graph
    x = []
    y = []
    for i in range(len(close_prices)):
        x.append(i+1)
        y.append(float(close_prices[i]))
    return x,y

def plot_price_weekly(n_weeks): #plotting price movement graph
    x,y = price_mvt_weekly(n_weeks)
    plt.figure()
    plt.plot(x,y)
    plt.title("Weekly Price Movement")
    plt.xlabel("Week")
    plt.ylabel("Price")
    plt.legend(["STOCK_US_XNAS_AAPL"],fontsize="10")
    plt.savefig("price_movement_weekly.png")

def log_return_weekly(n_weeks): #re-computing y for logreturn graph
    x,y = price_mvt_weekly(n_weeks)
    x_lr = x[1:]
    y_lr = []
    for j in range(1,len(y)):
        lr = np.log(y[j]/(y[j-1]))
        y_lr.append(lr)
    return x_lr,y_lr

def plot_logreturn_weekly(n_weeks): #plotting logreturn graph
    x,y = log_return_weekly(n_weeks)
    plt.figure()
    plt.plot(x,y)
    plt.title("Log Weekly Return")
    plt.xlabel("Week")
    plt.ylabel("Logreturn")
    plt.legend(["STOCK_US_XNAS_AAPL"],fontsize="10")
    plt.savefig("logreturn_weekly.png")

def hist_plot_weekly(n_bins, n_weeks): #plotting histogram + normal distribution curve
    lr = log_return_weekly(n_weeks)[1]
    x_norm = np.linspace(np.min(lr),np.max(lr),100)
    y_norm = 1/(np.std(lr)*np.sqrt(2*np.pi))*np.exp(-0.5*((x_norm-np.average(lr))/np.std(lr))**2)
    plt.figure()
    plt.hist(lr,bins=n_bins,edgecolor="black",density=True,label="Samples")
    plt.plot(x_norm,y_norm,'red',label="Normal distribution")
    plt.legend(fontsize='8')
    plt.title("Log Weekly Return Histogram")
    plt.xlabel("Returns")
    plt.ylabel("Density (%)")
    plt.savefig("Histogram_weekly.png")

def mean_and_variance_weekly(n_weeks):
    lr = log_return_weekly(n_weeks)[1]
    mean = np.mean(lr)
    variance = np.var(lr)
    return mean, variance

def qq_plot_weekly(n_weeks): #plotting quantile-quantile plot
    lr = log_return_weekly(n_weeks)[1]
    q_norm = np.percentile(np.random.normal(0,1,len(lr)), np.linspace(0,100,len(lr)))
    q_sample = np.percentile(lr, np.linspace(0,100,len(lr)))
    #fitting straight line
    m = np.std(q_sample) / np.std(q_norm)
    c = np.mean(q_sample) - m * np.mean(q_norm)
    plt.figure()
    plt.scatter(q_norm, q_sample)
    plt.plot(q_norm,(m*q_norm+c),color="red")
    plt.title("Quantile-Quantile Plot")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.savefig("QQ_Plot_weekly.png")


plot_price_weekly(len(close_prices))
plot_logreturn_weekly(len(close_prices))
hist_plot_weekly(10,len(close_prices))
print(mean_and_variance_weekly(len(close_prices)))
qq_plot_weekly(len(close_prices))