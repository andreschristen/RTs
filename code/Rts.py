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

Go directly to __main__ for examples.

"""

import os
from datetime import date, timedelta
from pickle import load, dump

from numpy import arange, diff, loadtxt, zeros, flip, array, log, quantile, ones
from numpy import savetxt, linspace, exp, cumsum, where, append, sqrt
from numpy import sum as np_sum
from scipy.stats import erlang, gamma, nbinom, uniform, beta
from scipy.stats import t as t_student
from matplotlib.pyplot import subplots, rcParams, close
from matplotlib.dates import drange

from pytwalk import pytwalk

from plotfrozen import PlotFrozenDist


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
    
    if type(data_fnam) == str:
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


"""
def loglikelihood_NB( x, mu, psi):
    mu_psi = mu/psi
    return -gammaln(x + 1) + gammaln(x + psi) - gammaln(psi)\
            -(x + psi)*log(1 + mu_psi) + x*log(mu_psi)
"""
def loglikelihood_NB( x, mu, psi):
    return beta.logcdf(x, mu*psi, (1-mu)*psi)


def Rts_NB( data, n=30, tau=7, psi=10, IP_dist=erlang( a=3, scale=8/3),\
            Rt_pr_a=5, Rt_pr_b=5/5, q=[10,25,50,75,90]):
    """Calculate Rt Using a Negative Binomial instead of Poisson.
       Here one needs to fix psi = 1/theta (= 10).
    
    
        Extension of (not documented):
        
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
        If q ia a single integer, return a simulation of the Rts, for each Rt
       
       Returns: a (len(q), n) array with quantiles of the R_{t,\tau}'s.
    """
    
    if isinstance( q, list): ## Return a list of quantiles
        q = array(q)/100
        quantiles = zeros(len(q))
        rt = zeros(( len(q), n))
        simulate = False
    else: ## If q ia a single integer, return a simulation of the Rts of size q, for each Rt
        rt = zeros(( q, n))
        simulate = True
       

    m = len(data)
    w = diff(IP_dist.cdf( arange( 0, m+1)))
    w /= sum(w)
    w = flip(w)
    
    R = linspace( 0.1, 3.0, num=100)
    DeltaR = R[1]-R[0]
    
    #omega = 1
    #theta = THETA_MEAN #0.01
    #psi = 1/theta        

    #fig, axs = subplots(nrows=5, ncols=1, figsize=( 5, 5))
    for t in range(max(m-n,0), m):
        #S1 = 0.0
        log_likelihood_I = zeros(R.shape) ## Same size of array for values for R
        if sum(data[:t]) <= 10:# Only for more than 10 counts
            continue
        for k in range(tau):
            I = data[:(t-k)] ## window of reports
            Gammak = I @ w[(m-(t-k)):] #\Gamma_k
            #S1 += Gammak
            I_k = data[(t-k)]
            log_likelihood_I += loglikelihood_NB( I_k, R*Gammak, psi)
        log_post = log_likelihood_I + gamma.logpdf( R, Rt_pr_a, scale=1/Rt_pr_b)
        pdf = exp(log_post)
        pdf /= sum(pdf)*DeltaR
        cdf = cumsum(pdf)*DeltaR
        if simulate:
            u = uniform.rvs()
            rt[:,t-(m-n)] = R[where(cdf < u)[0][-1]]
        else:
            for i,qua in enumerate(q):
                quantiles[i] = R[where(cdf < qua)[0][-1]]
            rt[:,t-(m-n)] = quantiles
    return rt


def PlotRts_NB( data_fnam, init_date, psi, trim=0,\
             tau=7, n=30, IP_dist=erlang( a=3, scale=8/3), Rt_pr_a=5, Rt_pr_b=5/5,\
             q=[10,25,50,75,90], csv_fnam=None, color='blue', ax=None):
    """Makes a board with the Rt evolution for the past n days (n=30).
       All parameters are passed to function Rts_NB.
       csv_fnam is an optional file name toi save the Rts info.
       ax is an Axis hadle to for the plot, if None, it creates one and retruns it.
    """
    
    if type(data_fnam) == str:
        data = loadtxt(data_fnam)
    else:
        data = data_fnam.copy()
        data_fnam = " "
    if trim < 0:
        data = data[:trim,:]

    rts = Rts_NB(data=data[:,1],\
             tau=tau, psi=psi, n=n, IP_dist=IP_dist, q=q,\
             Rt_pr_a=Rt_pr_a, Rt_pr_b=Rt_pr_b)
 
    m = data.shape[0]
    last_date = init_date + timedelta(m)
    if ax == None:
        fig, ax = subplots(figsize=( n/3, 3.5) )
    for i in range(n):
        h = rts[:,i]
        ax.bar( x=i, bottom=h[0], height=h[4]-h[0], width=0.9, color=color, alpha=0.25)
        ax.bar( x=i, bottom=h[1], height=h[3]-h[1], width=0.9, color=color, alpha=0.25)
        ax.hlines( y=h[2], xmin=i-0.9/2, xmax=i+0.9/2, color='red' )
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




class Rts_NB_psi:
    def __init__( self, data_fnam, init_date, trim=0, tau=7, n=30, IP_dist=erlang( a=3, scale=8/3),\
                Rt_pr_a=5, Rt_pr_b=5/5, q=[10,25,50,75,90], workdir="./../"):
        """Calculate Rt Using a Negative Binomial with unknown psi = 1/theta.
            Here one needs to run the MCMC first, RunMCMC.
            See example below.    
    
            Extension of (not documented): 
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
           
        """
        
        self.data_fnam = data_fnam
        data = loadtxt(workdir + 'data/' + data_fnam + '.csv')
        self.workdir = workdir
        if trim < 0:
            self.data = data[:trim,1]
        else:
            self.data = data[:,1]
        #convolve
        self.init_date = init_date
        self.m = len(data)
        self.IP_dist = IP_dist
        self.w = diff(IP_dist.cdf( arange( 0, self.m+1)))
        self.w /= sum(self.w)
        self.w = flip(self.w)
        self.n = min(self.m, n)
        self.tau = tau
        self.Rt_pr_a = Rt_pr_a
        self.Rt_pr_b = Rt_pr_b
        self.prior = gamma( self.Rt_pr_a, scale=1/self.Rt_pr_b)
        #omega = 1
        self.psi = 100
        self.psi_prior = gamma( 3, scale=self.psi/3)
        for t in range( self.m - self.n, self.m):
            if sum(self.data[:t]) <= 10:# Rt calculated only for more than 10 counts
                print("Not more than 10 counts for day %d" % (-t,))
                self.n -= 1
        self.Gammak = zeros(self.m) ##We calculate all gammas previously:
        for s in range(self.m):
            self.Gammak[s] = self.data[:s] @ self.w[(self.m-s):] #\Gamma_k

        if os.path.isfile(workdir + 'output/' + self.data_fnam + '_rts.pkl'): # samples file exists
            print("File with rts and psi samples exists, loading rts ...", end=' ')
            self.rts = load(open(workdir + 'output/' + self.data_fnam + '_rts.pkl', 'rb'))
            self.psi_samples = load(open(workdir + 'output/' + self.data_fnam + '_rts_psi.pkl', 'rb'))
        else:
            print("File with rts and psi samples does not exist, run RunMCMC first.")
        
    def logpost( self, Rs, psi):
        log_post = 0.0
        for t in range( self.m - self.n, self.m):
            log_post += self.prior.logpdf( Rs[t-(self.m - self.n)]) +\
                np_sum(loglikelihood_NB( self.data[(t-self.tau+1):t], Rs[t-(self.m - self.n)]*tst.Gammak[(t-self.tau+1):t], psi))
            #log_post += sum([loglikelihood_NB( self.data[s], Rs[t-(self.m - self.n)]*self.Gammak[s], psi) for s in range( t-self.tau+1, t)])
            """
            for k in range(self.tau):
                s = t-k
                #I = self.data[:s] ## window of reports
                #Gammak = self.data[:s] @ self.w[(self.m-s):] #\Gamma_k
                #I_k = self.data[s]
                log_post += loglikelihood_NB( self.data[s], Rs[t-(self.m - self.n)]*self.Gammak[s], psi)
            log_post += self.prior.logpdf( Rs[t-(self.m - self.n)])
            """
        return log_post
    
    def sim_init(self):
        """Simulate initial values from the Rts_NB and the prior for psi."""
        # Shake the Rts_NB simulation to avoid repeated values
        #shake = Rts_NB( self.data*self.Z, tau=self.tau, n=self.n, IP_dist=self.IP_dist,\
        #    Rt_pr_a=self.Rt_pr_a, Rt_pr_b=self.Rt_pr_b, q=1) + 0.001*uniform.rvs(size=self.n)
        shake = ones(self.n) + 0.001*uniform.rvs(size=self.n)
        return append( shake, self.psi_prior.rvs(size=1))
        #Simulate intial values from the prior.
        #return append(self.prior.rvs(size=self.n),self.psi_prior.rvs(size=1))
    
    def support(self, x):
        rt = all( (0.1 <= x[:-1]) * (x[:-1] <= 40) ) #Rt's
        rt &= (x[-1] > 0.0)
        return rt
    
    def RunMCMC( self, T, burnin=5000, q=[10,25,50,75,90]):
        """Run twalk MCMC, T = number of iterations.
           burnin, thining = IAT.
        """

        #self.twalk = pytwalk(n = self.n+1, U=lambda x: -self.logpost( x[:-1], self.psi), Supp =self.support) #Ignore x[-1] = psi    
        self.twalk = pytwalk(n = self.n+1, U=lambda x: -self.logpost( x[:-1], x[-1]) - self.prior.logpdf(x[-1]), Supp =self.support)    
        self.twalk.Run( T=T, x0 = self.sim_init(), xp0 = self.sim_init())
        self.burnin = burnin
        self.Rts(q=q)
        dump( self.rts, open(self.workdir + 'output/' + self.data_fnam + '_rts.pkl', 'wb'))
        self.psi_samples = self.twalk.Output[self.burnin:, self.n]
        dump( self.psi_samples, open(self.workdir + 'output/' + self.data_fnam + '_rts_psi.pkl', 'wb'))
    
    def PlotPostPsi( self, ax=None):
        if ax == None:
            fig, ax = subplots(figsize=( 5,5) )
        PlotFrozenDist(self.psi_prior, color='green', ax=ax)
        ax.hist( self.psi_samples, density=True)
        ax.set_xlabel(r'$\psi$')

    def PlotPostRt( self, i, ax=None):
        if ax == None:
            fig, ax = subplots(figsize=( 5,5) )
        #PlotFrozenDist(self.psi_prior, color='green', ax=ax)
        ax.hist( self.twalk.Output[self.burnin:,i], density=True)
        ax.set_xlabel(r'$R_%d$' % (i))

    def Rts(  self, q=[10,25,50,75,90]):

        if isinstance( q, list): ## Return a list of quantiles
            q = array(q)/100
            rts = zeros(( len(q), self.n))
            simulate = False
        else: ## If q ia a single integer, return a simulation of the Rts of size q, for each Rt
            rts = zeros(( q, self.n))
            simulate = True
        self.q = q
        self.simulate = simulate
        #fig, axs = subplots(nrows=5, ncols=1, figsize=( 5, 5))
        for i in range(self.n):
            if simulate:
                #u = uniform.rvs()
                rts[:,i] = self.twalk.Output[self.burnin+0,i]
            else:
                rts[:,i] = quantile( self.twalk.Output[self.burnin:,i], q=q)
        self.rts = rts
        return rts
    
    def PlotRts( self, color='blue', median_color='red', csv_fnam=None, ax=None):
        """Makes a board with the Rt evolution.
           csv_fnam is an optional file name to save the Rts info.
           ax is an Axis hadle to for the plot, if None, it creates one and retruns it.
        """
        
        #self.rts already been produced after running RunMCMC
        last_date = self.init_date + timedelta(self.m)
        
        if ax == None:
            fig, ax = subplots(figsize=( self.n/3, 3.5) )

        for i in range(self.n):
            h = self.rts[:,i]
            ax.bar( x=i, bottom=h[0], height=h[4]-h[0], width=0.9, color=color, alpha=0.25)
            ax.bar( x=i, bottom=h[1], height=h[3]-h[1], width=0.9, color=color, alpha=0.25)
            ax.hlines( y=h[2], xmin=i-0.9/2, xmax=i+0.9/2, color=median_color )
        ax.set_title(self.data_fnam + r", $R_t$, dist. posterior.")
        ax.set_xlabel('')
        ax.set_xticks(range(self.n))
        ax.set_xticklabels([(last_date-timedelta(self.n-i)).strftime("%d.%m") for i in range(self.n)], ha='right')
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
            days = drange( last_date-timedelta(self.n), last_date, timedelta(days=1))
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
            savetxt( csv_fnam, sv, delimiter=', ', fmt='%.1f', header="year, month, day, " + q_str, comments='')
        return ax


class Rts_AR:
    def __init__( self, data_fnam, init_date, trim=0,\
                IP_dist=erlang( a=3, scale=8/3), tau=7, m0=0, c_a_0=1, w_a_t=2/7, n0=2, s0=3,\
                n=30, pred=0, workdir="./../"):
        """Calculate Rt Using a log autoregressive time series on the logs.
    
           See: ...

            See example below.
        
           Parameters:
               
           data_fnam: file name = workdir + 'data/' + data_fnam + '.csv'
           or array with case incidence.
           init_date: intial date for firt datum, e.g. date(2020, 2, 27).
           trim: (negative) cut trim days at the end of data.
           
           tau: number of days to lern form the past (default 7, see paper).
           n: calculate n R_t's to the past n days (default 30).
           IP_dist: 'frozen' infectiousness profile distribution,
               default erlang( a=3, scale=8/3), chosen for covid19.
           Only the cdf is needed, ie. IP_dist.cdf(i), to calculate w_s.

           m0=0, c_a_0=1, w_a_t=0.25, n0=2, s0=3, m_0, c_0^*, w_t^*, n_0 prior
               hyperparameters (see paper).
        """
        
        self.data_fnam = data_fnam
        data = loadtxt(workdir + 'data/' + data_fnam + '.csv')
        self.workdir = workdir
        if trim < 0:
            self.data = data[:trim,1]
        else:
            self.data = data[:,1]
        self.init_date = init_date
        self.m = len(self.data) ##Data size
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
                
    def PlotPostRt( self, i, ax=None, color='black'):
        """Plot the i-th Rt posterior distribution."""
        if ax == None:
            fig, ax = subplots(figsize=( 5,5) )
        t = i+self.tau
        y = linspace( 0.01, 4, num=500)
        ### Transformed pdf using the Jacobian y^{-1}
        pdf = (y**-1) * t_student.pdf( log(y), df=self.hiper[t+1,0], loc=self.hiper[t+1,5], scale=sqrt(self.hiper[t+1,6]) )
        ax.plot( y, pdf, '-', color=color)
        ax.set_ylabel("Density")
        ax.set_xlabel(r'$R_{%d}$' % (i))

    def PlotRts( self, color='blue', median_color='red', x_jump=1, plot_area=[0.4,2.2], alpha=0.25, csv_fnam=None, ax=None):
        """Makes a board with the Rt evolution.
           csv_fnam: optional file name to save the Rts info: workdir/csv/csv_fnam.csv
           ax: Axis hadle to for the plot, if None, it creates one and retruns it.
           x_jump: put ticks every x_jump days.
           plot_area: ([0.4,2.2]), interval with the y-axis (Rt values) plot area. 
       """
        
        #self.rts already been produced after running CalculateRts
        last_date = self.init_date + timedelta(self.m)
        
        if ax == None:
            fig, ax = subplots(figsize=( self.n/3, 3.5) )

        ### Plot the Rt's posterior quantiles
        for i in range(self.n):
            h = self.rts[:,i]
            ax.bar( x=i, bottom=h[0], height=h[4]-h[0], width=0.9, color=color, alpha=0.25)
            ax.bar( x=i, bottom=h[1], height=h[3]-h[1], width=0.9, color=color, alpha=0.25)
            ax.hlines( y=h[2], xmin=i-0.9/2, xmax=i+0.9/2, color=median_color )
        ### Plot the observed Rt's
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
        ax.set_xticklabels([(last_date-timedelta(self.n-i)).strftime("%d.%m") for i in range(0,self.n,x_jump)], ha='right')
        ax.tick_params( which='major', axis='x', labelsize=10, labelrotation=30)
        ax.axhline(y=1, color='green')
        ax.axhline(y=2, color='red')
        ax.axhline(y=3, color='darkred')
        ax.set_ylim(plot_area)
        ax.set_yticks(arange( plot_area[0], plot_area[1], step=0.2))
        ax.tick_params( which='major', axis='y', labelsize=10)
        ax.grid(color='grey', linestyle='--', linewidth=0.5)
        #fig.tight_layout()
        if csv_fnam != None:
            days = drange( last_date-timedelta(self.n), last_date, timedelta(days=1))
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
            savetxt( self.workdir + "csv/" + csv_fnam + ".csv", sv, delimiter=', ', fmt='%.1f', header="year, month, day, " + q_str, comments='')
        return ax


##### Dirctionary with general information for the metro zone or region to be analyzed:
#####     id               Name   not used  Population   init date                     
ZMs = {  "9-01":         ["Mexico city", 2, 21.942666e6, date(2020, 2, 27)],\
         "15-02":        ["Toluca",      1,  2.377828e6, date(2020, 3,  7)],\
         "31-01":        ["Mérida",      2,  1.237697e6, date(2020, 3,  7)],\
         "17-02":        ["Cuernavaca",  1,  1.059521e6, date(2020, 3,  2)],\
         "12-01":        ["Acapulco",    2,  0.919726e6, date(2020, 3, 11)],\
         "25-01":        ["Culiacán",    2,  0.962871e6, date(2020, 3,  1)],\
         "23-01":        ["Cancun",      2,  0.867768e6, date(2020, 3,  1)]}

### The correponding data files have two columns separated by space, deaths and incidence.
### Each row is one day.
### The file for clave="9-01" (Mexico city) is: ../data/clave.csv etc.


if __name__=='__main__':

    rcParams.update({'font.size': 14})
    close('all')
    #Plot the imputed serial time distribution for covid: erlang( a=3, scale=8/3 )
    fig, ax = subplots( num=30, figsize=( 4.5, 3.5))
    PlotFrozenDist( erlang( a=3, scale=8/3 ), ax=ax)
    ### Plota the erlang( a=5, scale=9/5 ) alternative
    PlotFrozenDist( erlang( a=5, scale=9/5 ), color='grey', ax=ax)
    ax.set_xlim((0,20))
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_ylabel(r"Density")
    ax.set_xlabel("days")
    ax.set_title("")
    fig.tight_layout()
    fig.savefig("../figs/Covid19_SerialTimeDist.png")

    ### Plot the Rt's estimation.  Only Merida, '13-01'  and Mexico city, '9-01', are in the paper
    claves = ['15-02', '17-02', '23-01', '25-01', '12-01', "31-01", '9-01'] 
    n=60  ## Number of days to calculate the Rt's
    trim=0 ## Number of days to cut data from the end, negative, e.g. -10, cut 10 days
    x_jump = 7 ## For ploting, put ticks every x_jump days.

    for i,clave in enumerate(claves):
        print(clave)
        ### Open an instance of the Rts_AR class: 
        tst = Rts_AR( clave, init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=5, n=n)
        tst.CalculateRts() # Most be called before ploting the Rt's
        ### Plot the Rts:
        fig, ax = subplots( num=i+1, figsize=( 8, 3.5))
        ### Plot Cori et al (2013) Poisson model version:
        PlotRts_P( '../data/%s.csv' % (clave,), init_date=ZMs[clave][3]+timedelta(days=4),\
                n=tst.n, trim=trim, ax=ax, color='green', alpha=0.5, median_color='black')
        ### Plot ours:
        tst.PlotRts( ax=ax, x_jump=x_jump, plot_area=[0.4,2.2], csv_fnam=clave)
        ax.set_title("")
        ax.set_ylabel(r"$R_t$")
        ax.set_xlabel("")
        ax.set_title(ZMs[clave][0] + ", Mexico")
        fig.tight_layout()
        fig.savefig("../figs/%s_Rts_AR.png" % (clave,))
        if clave == '9-01':
            m_max = tst.m
    ax.set_xlabel("day.month, 2020")
    fig.tight_layout()
    fig.savefig("../figs/%s_Rts_AR.png" % (clave,))

    ### Figure with Cori et al (2013) posterior distributions of '31-01' and '9-01'
    fig1, ax1 = subplots( num=20, nrows=1, ncols=2, figsize=( 10, 3.5))
    color = [ "red", "black", "darkred"]
    for i,clave in enumerate([ '31-01', '9-01']):        
        tst = Rts_AR( clave, init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
        a, b = Rts_P( tst.data, tau=7, n=30, q=2)
        ax1[0].plot( arange(m_max-tst.m, m_max, 1), tst.data, '.-', color=color[i], label=ZMs[clave][0])
        PlotFrozenDist( gamma( a[-1], scale=b[-1]), ax=ax1[1], color=color[i])
    last_date = tst.init_date + timedelta(tst.m)
    ax1[0].set_xlabel('')
    ax1[0].set_xticks(range(0,tst.m,x_jump*2))
    ax1[0].set_xticklabels([(last_date-timedelta(tst.m-i)).strftime("%d.%m") for i in range(0,tst.m,x_jump*2)], ha='right')
    ax1[0].tick_params( which='major', axis='x', labelsize=10, labelrotation=30)    
    ax1[0].set_xlabel("day.month, 2020")
    #ax1[0].set_ylim((0,1.1*max(tst.data[-n:])))
    ax1[0].grid(color='grey', linestyle='--', linewidth=0.5)
    ax1[0].set_ylabel(r"Incidence")
    ax1[0].legend(loc=0, shadow = False)

    ### Add '31-01', with incidence multiplied by 10
    clave = '31-01'
    tst = Rts_AR( clave, init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
    a, b = Rts_P( tst.data*10, tau=7, n=30, q=2)
    ax1[0].plot( arange(m_max-tst.m, m_max, 1), tst.data*10, '.-', color=color[2])
    PlotFrozenDist( gamma( a[-1], scale=b[-1]), ax=ax1[1], color=color[2])
    ax1[1].set_xticks(arange(0.8,1.4,0.2))
    ax1[1].set_xlabel(r"$R_t$, " + (last_date-timedelta(1)).strftime("%d.%m.%Y")) 
    ax1[1].grid(color='grey', linestyle='--', linewidth=0.5)

    fig1.tight_layout()
    fig1.savefig("../figs/Rts_Compare.png")

    ### Comparison of results changing the serial time distribution
    fig, ax = subplots( num=31, figsize=( 4.5, 3.5))
    tst = Rts_AR( clave, init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
    tst.CalculateRts()
    tst.PlotPostRt( i=n, ax=ax)
    #### Here we change the serial time:    Any other positive density could be used.
    tst = Rts_AR( clave, IP_dist=erlang( a=5, scale=9/5), init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
    tst.CalculateRts()
    tst.PlotPostRt( i=n, ax=ax, color='grey')
    ax.set_xlim((0.5,2.5))
    ax.set_xlabel(r"$R_t$, " + (last_date-timedelta(1)).strftime("%d.%m.%Y"))
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_title("")
    fig.tight_layout()
    fig.savefig("../figs/%s_Rts_Compare.png" % (clave,))

    """
    ################# Example of use of Rts_NB_psi and Rts_NB (not documented)
    T=100000
    for clave in claves: #Instance of the object and run the MCMC
        tst = Rts_NB_psi( clave, init_date=ZMs[clave][3], n=n)
        if T > 0:
            tst.RunMCMC(T=T)
        ### Plot the Rts
        close(1)
        fig, ax = subplots( num=1, figsize=( 10, 3.5) )
        tst.PlotRts( ax=ax)
        ax.set_title( ZMs[clave][0] + r", $R_t$ NB_psi.")
        fig.savefig("../figs/%s_Rts_NB_psi.png" % (clave,))
        ### Plot the posterior distribution of \psi
        close(3)
        fig, ax = subplots( num=3, figsize=( 5,5) )
        tst.PlotPostPsi(ax=ax)
        ax.set_title(ZMs[clave][0])
        fig.savefig("../figs/%s_Rts_NB_Post_psi.png" % clave)
        ### Fix \psi with the postrior expeted value and use that for PlotRts_NB
        close(2)
        fig, ax = subplots( num=2, figsize=( 10, 3.5) )
        psi = mean(tst.psi_samples) #Posterior mean of psi
        PlotRts_NB( '../data/%s.csv' % (clave,), init_date=ZMs[clave][3],\
                n=n, psi=psi, ax=ax)
        ax.set_title( ZMs[clave][0] + r", $R_t$ NB, fixed $\psi$.")
        fig.savefig("../figs/%s_Rts.png" % (clave,))
        
    """
    


