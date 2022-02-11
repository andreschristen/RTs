#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:28:54 2020

@author: Dr J A Christen (CIMAT-CONACYT, Mexico) jac at cimat.mx

Instantaneous reproduction numbers calculations.

Rts_P, Implementation of Cori et al (2013)

Rts_AR, new filtering version using an autoregressive linear model of Capistrán, Capella and Christen (2020):
https://arxiv.org/abs/2012.02168, 05DIC2021 

01FEB2021: Some buggs were corrected to avoid error when too low counts are used and for prediction when g=1.

Go directly to SimpleExample.py or to PaperPlots.py,
to reproduce all the plots (an more) in the paper.

"""

from datetime import date, timedelta

from numpy import arange, diff, loadtxt, zeros, flip, array, log, ones, ndarray
from numpy import savetxt, linspace, exp, sqrt
from scipy.stats import erlang, gamma
from scipy.stats import t as t_student
from matplotlib.pyplot import subplots
from matplotlib.dates import drange




def Rts_P( data, tau=7, n=30, IP_dist=erlang( a=3, scale=8/3),\
            Rt_pr_a=5, Rt_pr_b=5/5, q=[10,25,50,75,90]):
    """Calculate Rt as in: 
       Anne Cori, Neil M. Ferguson, Christophe Fraser, Simon Cauchemez,
       A New Framework and Software to Estimate Time-Varying Reproduction Numbers
       During Epidemics, American Journal of Epidemiology,
       Volume 178, Issue 9, 1 November 2013, Pages 1505–1512,
       https://doi.org/10.1093/aje/kwt133 
        
       data: array with case incidence.
       tau: Use a window tau (default 7) to calculate R_{t,\tau}'s.
       n: calculate n R_{t,\tau}'s to the past n days (default 30).
       IP_dist: 'frozen' infectiousness profile distribution,
           default erlang( a=3, scale=8/3), chosen for covid19.
           Only the cdf is needed, ie. IP_dist.cdf(i), to calculate w_s.
       Rt_pr_a=5, Rt_pr_b=5/5, parameters for the gamma prior for R_t.
       q=[10,25,50,75,90], quantiles to use to calulate in the post. dust for R_t.
        If q ia a single integer, return a simulation of the Rts of size q, for each Rt
       
       Returns: a (len(q), n) array with quantiles of the R_{t,\tau}'s.
    """
    
    if isinstance( q, list): ## Return a list of quantiles
        q = array(q)/100
        rt = zeros(( len(q), n))
        simulate = False
    else: ## If q ia a single integer, return a simulation of the Rts of size q, for each Rt
        if q == 2: # return a and b of post gamma
            rt = zeros(( q, n))
        else:
            rt = zeros(( q, n))
        simulate = True
       

    m = len(data)
    w = diff(IP_dist.cdf( arange( 0, m+1)))
    w /= sum(w)
    w = flip(w)
    
    for t in range(max(m-n,0), m):
        S1 = 0.0
        S2 = 0.0
        if sum(data[:t]) <= 10:# Only for more than 10 counts
            continue
        for k in range(tau):
            I = data[:(t-k)] ## window of reports
            S2 += data[(t-k)]
            S1 += sum(I * w[(m-(t-k)):]) #\Gamma_k
        #print( (Rt_pr_a+S2) * (1/(S1 + 1/Rt_pr_b)), (Rt_pr_a+S2), 1/(S1 + 1/Rt_pr_b))
        if simulate:
            if q == 2: #Return Rt_pr_a+S2, scale=1/(S1 + 1/Rt_pr_b)
                rt[:,t-(m-n)] = Rt_pr_a+S2, 1/(S1 + 1/Rt_pr_b)
            else:
                rt[:,t-(m-n)] = gamma.rvs( Rt_pr_a+S2, scale=1/(S1 + 1/Rt_pr_b), size=q)
        else:
            rt[:,t-(m-n)] = gamma.ppf( q, Rt_pr_a+S2, scale=1/(S1 + 1/Rt_pr_b))
    return rt




def PlotRts_P( data_fnam, init_date, trim=0,\
             tau=7, n=30, IP_dist=erlang( a=3, scale=8/3), Rt_pr_a=5, Rt_pr_b=5/5,\
             q=[10,25,50,75,90], csv_fnam=None, color='blue', median_color='red', alpha=0.25, ax=None):
    """Makes a board with the Rt evolution for the past n days (n=30).
       All parameters are passed to function Rts_P.
       csv_fnam is an optional file name toi save the Rts info.
       ax is an Axis hadle to for the plot, if None, it creates one and retruns it.
    """
    
    if not isinstance(data_fnam, ndarray):
        data = loadtxt(data_fnam)
    else:
        data = data_fnam.copy()
        data_fnam = " "

    if trim < 0:
        data = data[:trim,:]

    rts = Rts_P(data=data[:,1],\
             tau=tau, n=n, IP_dist=IP_dist, q=q,\
             Rt_pr_a=Rt_pr_a, Rt_pr_b=Rt_pr_b)
 
    m = data.shape[0]
    last_date = init_date + timedelta(m)
    if ax == None:
        fig, ax = subplots(figsize=( n/3, 3.5) )
    for i in range(n):
        h = rts[:,i]
        ax.bar( x=i, bottom=h[0], height=h[4]-h[0], width=0.9, color=color, alpha=alpha)
        ax.bar( x=i, bottom=h[1], height=h[3]-h[1], width=0.9, color=color, alpha=alpha)
        ax.hlines( y=h[2], xmin=i-0.9/2, xmax=i+0.9/2, color=median_color )
    ax.set_title(data_fnam + r", $R_t$, dist. posterior.")
    ax.set_xlabel('')
    ax.set_xticks(range(n))
    ax.set_xticklabels([(last_date-timedelta(n-i)).strftime("%d.%m") for i in range(n)], ha='right')
    ax.tick_params( which='major', axis='x', labelsize=10, labelrotation=30)
    ax.axhline(y=1, color='green')
    ax.axhline(y=2, color='red')
    ax.axhline(y=3, color='darkred')
    ax.set_ylim((0.5,3.5))
    ax.set_yticks(arange( 0.4, 3.4, step=0.2))
    ax.tick_params( which='major', axis='y', labelsize=10)
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    #fig.tight_layout()
    if csv_fnam != None:
        days = drange( last_date-timedelta(n), last_date, timedelta(days=1))
        ### To save all the data for the plot, 
        ### columns: year, month, day,  q_05, q_25, q_50, q_75, q_95
        ###             0      1   2       3     4     5     6     7          
        sv = -ones(( len(days), 3+len(q)))
        for i,day in enumerate(days):
            d = date.fromordinal(int(day))
            sv[ i, 0] = d.year
            sv[ i, 1] = d.month
            sv[ i, 2] = d.day
            sv[ i, 3:] = rts[:,i]
        q_str = ', '.join(["q_%02d" % (qunt,) for qunt in q])
        savetxt( csv_fnam, sv, delimiter=', ', fmt='%.1f', header="year, month, day, " + q_str, comments='')
    return ax


class Rts_AR:
    """
       Calculate Rt Using a log autoregressive time series.
    
       See: https://arxiv.org/abs/2012.02168, 05DIC2021 
    
       Go directly to SimpleExample.py for basic functionaloty
       or to PaperPlots.py, to reproduce all the plots (an more) in the paper.

       Parameters:
       
       data_fnam: file name for the case incidence  = 
           workdir + data_fnam + '.csv'
           or a numpy array with case incidence.
       workdir: default "./", current local dir.
       
       init_date: intial date for first datum, e.g. date(2020, 2, 27).
       col: which column in the data to use (default 1, cases, col 0 is deaths
           in the examples data files, ignored if data_fnam is an array)
       trim: if negative, cut |trim| days at the end of data, (default 0).
       n: calculate n R_t's to the past n days (default 30).

       tau: number of days to lern form the past (default 7, see paper).
       IP_dist: 'frozen' infectiousness profile distribution,
          default erlang( a=3, scale=8/3), chosen for covid19.
          Only the cdf is needed, ie. IP_dist.cdf(i), to calculate w_s.

       Prior hyperparameters (see paper):
       m0=0, c_a_0=1, w_a_t=0.25, n0=2, s0=3, m_0, c_0^*, w_t^*, n_0 prior
    """
    def __init__( self, data_fnam, init_date, col=1, trim=0,\
                IP_dist=erlang( a=3, scale=8/3), tau=7, m0=0, c_a_0=1, w_a_t=2/7, n0=2, s0=3,\
                n=30, pred=0, workdir="./"):
        
        if not isinstance(data_fnam, ndarray):
            self.data_fnam = data_fnam
            data = loadtxt(workdir + data_fnam + '.csv')
            self.workdir = workdir
            if trim < 0:
                self.data = data[:trim,col]
            else:
                self.data = data[:,col]
        else:
            data = data_fnam.copy()
            if trim < 0:
                self.data = data[:trim]
            else:
                self.data = data
            self.data_fnam = " "
            self.workdir = " "
        self.init_date = init_date
        self.m = len(self.data) ##Data size
        self.last_date = self.init_date + timedelta(self.m)
        ### Calculate the serial time distribution
        self.IP_dist = IP_dist
        self.w = diff(IP_dist.cdf( arange( 0, self.m+1)))
        self.w /= sum(self.w)
        self.w = flip(self.w)
        
        ### Calculation range 
        self.shift = 5*tau #Number of days to start calculation before the frist Rt. 
        self.n = min(self.m, n) #Number of Rt's to calculate, from the present into the past.
        self.N = n+self.shift #Total range (into the past) for calculation
        #If self.N is larger than the whole data set
        if self.N > (self.m-1):
            self.n -= self.N - (self.m-1)#Reduce self.n accordingly
            self.N = n+self.shift
            if self.n < 0:
                raise ValueError("ERROR: Not enough data to calculate Rts: 5*tau > %d (data size)" % (self.m,))
            print("Not enough data to calculate Rts: 5*tau + n > %d (data size)" % (self.m,))
            print("Reducing to n=%d" % (self.n,))
        for t in range(self.n):
            if self.data[self.m-(self.n - t)] >= 10:
                break
            else:
                self.n -= 1 #Reduce n if the counts have not reached 10
                print("Incidence below 10, reducing n to %d." % (self.n,))
        self.N = self.n+self.shift
        ### Setting prior parameters
        self.delta = 1-(1/tau)
        self.tau = tau
        self.pred = pred
        self.g = 1 #exp(-2/tau)
        self.m0 = m0
        self.c_a_0 = c_a_0
        self.w_a_t = w_a_t
        self.n0 = n0
        self.s0 = s0
        """
        ### Calculation range
        for t in range( self.m - self.N, self.m):
            if sum(self.data[:t]) <= 10:# Rt calculated only for more than 10 counts
                print("Not more than 10 counts for day %d" % (-t,))
                self.n -= 1
                self.N = min(self.m, n+self.shift)
        """
        ### We calculate all gammas previously: 
        self.Gammak = zeros(self.m) 
        for s in range(self.m):
            self.Gammak[s] = self.data[:s] @ self.w[(self.m-s):] #\Gamma_k
        ### Calculate the log data:
        ### We add 1e-6 for convinience, since very early data may be zero
        ### This makes no diference at the end.
        self.y = log(self.data + 1e-6) - log(self.Gammak + 1e-6)
        
        
    def sim_data( self, R, I0):
        pass
        
    def CalculateRts( self, q=[10,25,50,75,90]):
        """Calculate the posterior distribution and the Rt's quantiles.
           q=[10,25,50,75,90], quantiles to use to calulate in the post. dust for R_t.

            If q ia a single integer, return a simulation of the Rts of size q, for each Rt.
            If q=2, save the mean and dispersion parameter of the posterior for Rt

        """
        if isinstance( q, list): ## Return a list of quantiles
            q = array(q)/100
            self.rts = zeros(( len(q), self.n))
            self.rts_pred = zeros((len(q), self.pred))
            simulate = False
        else: ## If q ia a single integer, return a simulation of the Rts of size q, for each Rt
            self.rts = zeros(( q, self.n))
            self.rts_pred = zeros(( q, self.pred))
            simulate = True
        self.q = q
        self.simulate = simulate

        ###          nt, at, rt, qt, st, mt, ct # hiperparameters
        ###           0  1    2   3   4   5   6
        self.hiper = zeros(( self.N+1, 7))
        ###                    nt, at, rt,    qt,    st,      mt,    ct # hiperparameters
        self.hiper[0,:] = self.n0, -1, -1,   -1, self.s0, self.m0, self.s0*self.c_a_0
        
        for t in range( self.N ):
            r_a_t = self.g**2 * self.hiper[t,6] + self.w_a_t #r^*_t
            At = r_a_t/(r_a_t + 1)

            self.hiper[t+1,0] = self.delta*self.hiper[t,0] + 1 #nt
            self.hiper[t+1,1] = self.g * self.hiper[t,5] #at
            et = self.y[self.m-(self.N - t)] - self.hiper[t+1,1]
            self.hiper[t+1,2] = self.hiper[t,4]*r_a_t #rt
            self.hiper[t+1,3] = self.hiper[t,4]*(r_a_t + 1) #qt
            # st:
            self.hiper[t+1,4] = self.delta*(self.hiper[t,0]/self.hiper[t+1,0])*self.hiper[t,4] +\
                                self.hiper[t,4]/self.hiper[t+1,0] * (et**2/self.hiper[t+1,3])
            self.hiper[t+1,5] = self.hiper[t+1,1] + At*et #mt
            #ct
            self.hiper[t+1,6] = (self.hiper[t+1,4]/self.hiper[t,4]) * (self.hiper[t+1,2]- self.hiper[t+1,3]*At**2)

            if t >= self.shift:
                if self.simulate:
                   self.rts[:,t-self.shift] = exp(t_student.rvs( size=self.q, df=self.hiper[t+1,0], loc=self.hiper[t+1,5], scale=sqrt(self.hiper[t+1,6]) )) 
                else:
                   self.rts[:,t-self.shift] = exp(t_student.ppf( q=self.q, df=self.hiper[t+1,0], loc=self.hiper[t+1,5], scale=sqrt(self.hiper[t+1,6]) ))
        if self.pred>0:
            t = self.N
            self.pred_hiper = zeros(( self.pred, 2)) # a_t^k and r_t^k
            for k in range(self.pred):
                self.pred_hiper[k,0] = self.g**(k+1) * self.hiper[t,5] #a_t^k
                if self.g == 1:
                    self.pred_hiper[k,1] = self.g**(2*(k+1)) * self.hiper[t,6] + self.w_a_t * (k+1) #r_t^k
                else:
                    self.pred_hiper[k,1] = self.g**(2*(k+1)) * self.hiper[t,6] + self.w_a_t * ((1-self.g**(2*(k+1)))/(1-self.g**2))  #r_t^k
                    
                if self.simulate:
                   self.rts_pred[:,k] = exp(t_student.rvs( size=self.q, df=self.hiper[t,0], loc=self.pred_hiper[k,0], scale=sqrt(self.pred_hiper[k,1]) )) 
                else:
                   self.rts_pred[:,k] = exp(t_student.ppf( q=self.q,    df=self.hiper[t,0], loc=self.pred_hiper[k,0], scale=sqrt(self.pred_hiper[k,1]) ))
                
    def PlotPostRt( self, i, rng=(0.01,4), ax=None, linestyle='-', color='black', date_fmt="%Y.%m.%d", **kwargs):
        """Plot the i-th Rt posterior distribution, in the interval rng=(0.01,4)
           ax: if None, create an axes to plot (default), otherwise, use ax
           Ploting parameters for the density:
           linestyle='-', color='black', date_fmt="%Y.%m.%d"
           Any other ploting parameters to be passed to ax.plot:
               **kwargs
           
           The density is saved in self.rt and self.pdf, the method returns ax.
        """
        if ax is None:
            fig, ax = subplots(figsize=( 5,5) )
        t = self.shift + i
        self.rt = linspace( rng[0], rng[1], num=500)
        ### Transformed pdf using the Jacobian y^{-1}
        self.pdf = (self.rt**-1) * t_student.pdf( log(self.rt), df=self.hiper[t+1,0], loc=self.hiper[t+1,5], scale=sqrt(self.hiper[t+1,6]) )
        ax.plot( self.rt, self.pdf, linestyle=linestyle, color=color, **kwargs)
        ax.set_ylabel("Density")
        ax.set_xlabel(r"$R_t$, " + (self.last_date-timedelta(self.n-i)).strftime(date_fmt))
        return ax

    def PlotRts( self, color='blue', median_color='red', x_jump=1, plot_area=[0.4,2.2], alpha=0.25, plot_obs_rts=False,
                csv_fnam=None, ax=None):
        """Makes a board with the Rt evolution.
           csv_fnam: optional file name to save the Rts info: workdir/csv/csv_fnam.csv
           ax: Axis hadle to for the plot, if None, it creates one and retruns it.
           x_jump: put ticks every x_jump days.
           plot_area: ([0.4,2.2]), interval with the y-axis (Rt values) plot area. 
       """
        
        #self.rts already have been produced after running CalculateRts
        
        if ax is None:
            fig, ax = subplots(figsize=( self.n/3, 3.5) )

        ### Plot the Rt's posterior quantiles
        for i in range(self.n):
            h = self.rts[:,i]
            ax.bar( x=i, bottom=h[0], height=h[4]-h[0], width=0.9, color=color, alpha=0.25)
            ax.bar( x=i, bottom=h[1], height=h[3]-h[1], width=0.9, color=color, alpha=0.25)
            ax.hlines( y=h[2], xmin=i-0.9/2, xmax=i+0.9/2, color=median_color )
        ### Plot the observed Rt's
        if plot_obs_rts:
            ax.plot( exp(self.y[self.m-self.n:]), '-', color='grey')
        ### Plot the predictions
        if self.pred >0:
            for k in range(self.pred):
                h = self.rts_pred[:,k]
                i=self.n+k
                ax.bar( x=i, bottom=h[0], height=h[4]-h[0], width=0.9, color='light'+color, alpha=alpha)
                ax.bar( x=i, bottom=h[1], height=h[3]-h[1], width=0.9, color='light'+color, alpha=alpha)
                ax.hlines( y=h[2], xmin=i-0.9/2, xmax=i+0.9/2, color=median_color )
                
        ax.set_title(self.data_fnam + r", $R_t$, dist. posterior.")
        ax.set_xlabel('')
        ax.set_xticks(range(0,self.n,x_jump))
        ax.set_xticklabels([(self.last_date-timedelta(self.n-i)).strftime("%d.%m") for i in range(0,self.n,x_jump)], ha='right')
        ax.tick_params( which='major', axis='x', labelsize=10, labelrotation=30)
        ax.axhline(y=1, color='green')
        ax.axhline(y=2, color='red')
        ax.axhline(y=3, color='darkred')
        ax.set_ylim(plot_area)
        ax.set_yticks(arange( plot_area[0], plot_area[1], step=0.2))
        ax.tick_params( which='major', axis='y', labelsize=10)
        ax.grid(color='grey', linestyle='--', linewidth=0.5)
        #fig.tight_layout()
        if csv_fnam is not None:
            days = drange( self.last_date-timedelta(self.n), self.last_date, timedelta(days=1))
            ### To save all the data for the plot, 
            ### columns: year, month, day,  q_05, q_25, q_50, q_75, q_95
            ###             0      1   2       3     4     5     6     7          
            sv = -ones(( len(days), 3+len(self.q)))
            for i,day in enumerate(days):
                d = date.fromordinal(int(day))
                sv[ i, 0] = d.year
                sv[ i, 1] = d.month
                sv[ i, 2] = d.day
                sv[ i, 3:] = self.rts[:,i]
            q_str = ', '.join(["q_%02d" % (qunt,) for qunt in self.q])
            savetxt( self.workdir + "Rts_AR_csv/" + csv_fnam + ".csv", sv, delimiter=', ', fmt='%.1f', header="year, month, day, " + q_str, comments='')
        return ax



