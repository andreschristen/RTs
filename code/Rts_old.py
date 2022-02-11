#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:28:54 2020

@author: jac
"""

from datetime import date, timedelta

from numpy import arange, diff, loadtxt, zeros, flip, array, ones, savetxt
from scipy.stats import erlang, gamma
from matplotlib.pyplot import subplots, rcParams, close, hist
from matplotlib.dates import drange


def Rts( data, tau=7, n=30, IP_dist=erlang( a=3, scale=8/3),\
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




def PlotRts( data_fnam, init_date, trim=0,\
             tau=7, n=30, IP_dist=erlang( a=3, scale=8/3), Rt_pr_a=5, Rt_pr_b=5/5,\
             q=[10,25,50,75,90], csv_fnam=None, color='blue', median_color='red', ax=None):
    """Makes a board with the Rt evolution for the past n days (n=30).
       All parameters are passed to function Rts.
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

    rts = Rts(data=data[:,1],\
             tau=tau, n=n, IP_dist=IP_dist, q=q,\
             Rt_pr_a=Rt_pr_a, Rt_pr_b=Rt_pr_b)
 
    m = data.shape[0]
    last_date = init_date + timedelta(m)
    if ax == None:
        fig, ax = subplots(figsize=( n/3, 3.5) )
    for i in range(n):
        h = rts[:,i]
        ax.bar( x=i, bottom=h[0], height=h[4]-h[0], width=0.9, color=color, alpha=0.25)
        ax.bar( x=i, bottom=h[1], height=h[3]-h[1], width=0.9, color=color, alpha=0.25)
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
### Plot the infectiousness profile distribution, imputed for covid19
from plotfrozen import PlotFrozenDist
from scipy.stats import erlang
#Imputed serial time distribution: erlang
PlotFrozenDist( erlang( a=3, scale=8/3 ))
### Test:
from datetime import date
data = loadtxt('../data/9-01.csv')
rts = Rts(data=data[:,1]) # Reported cases
fig, ax = subplots(figsize=( 10, 3.5) )
PlotRts( '../data/9-01.csv', init_date=date( 2020, 2, 27), csv_fnam="9-01_Rts.csv", ax=ax)
ax.set_title( "Mexico City" + r", $R_t$, posterior dist.")
fig.savefig("9-01_Rts.png")
"""
