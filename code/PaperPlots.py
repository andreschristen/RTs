#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:03:41 2022

@author: jac
"""



from time import time
from datetime import date, timedelta

from numpy import arange

from scipy.stats import erlang, gamma, uniform
from matplotlib.pyplot import subplots, rcParams, close

from Rts_AR import Rts_AR, Rts_P, PlotRts_P
from plotfrozen import PlotFrozenDist



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


rcParams.update({'font.size': 14})
close('all')


### Comparison of results changing the serial time distribution
n=60  ## Number of days to calculate the Rt's
clave = '31-01'

#Plot the imputed serial time distribution for covid: erlang( a=3, scale=8/3 )
fig1, ax1 = subplots( num=30, figsize=( 4.5, 3.5))

trim=-10 ## Number of days to cut data from the end, negative, e.g. -10, cut 10 days
fig, ax = subplots( num=31, figsize=( 4.5, 3.5))
t0 = time()
for i in range(20):
    a = 3 + 4*uniform.rvs()-2
    mean = 8 + 14*uniform.rvs()-7 #5.20 (3.78 - 6.78)
    scale = mean/a
    ### Plota the erlang( a=5, scale=9/5 ) alternative
    PlotFrozenDist( gamma( a=a, scale=scale ), ax=ax1, color='grey', alpha=0.7)

    tst = Rts_AR( clave, workdir="./../data/", init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
    tst.CalculateRts()
    tst.PlotPostRt( i=n-1, ax=ax)
    #### Here we change the serial time:    Any other positive density could be used.
    tst = Rts_AR( clave, workdir="./../data/", IP_dist=gamma( a=a, scale=scale), init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
    tst.CalculateRts()
    tst.PlotPostRt( i=n-1, ax=ax, color='grey', alpha=0.7)
print("Time to calculate 20 posteriors: %f s." % (time()-t0,))
### Analysis with the assumed COVID19 Serial interval
PlotFrozenDist( erlang( a=3, scale=8/3 ), ax=ax1)
tst = Rts_AR( clave, workdir="./../data/", init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
tst.CalculateRts()
tst.PlotPostRt( i=n-1, ax=ax)
tst = Rts_AR( clave, workdir="./../data/", IP_dist=gamma( a=3, scale=8/3), init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
tst.CalculateRts()
tst.PlotPostRt( i=n-1, ax=ax, color='black')
### Analysis with a very different Serial interval
PlotFrozenDist( erlang( a=30, scale=20/30 ), ax=ax1, linestyle='--', color='green')
tst = Rts_AR( clave, workdir="./../data/", init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
tst.CalculateRts()
tst.PlotPostRt( i=n-1, ax=ax)
tst = Rts_AR( clave, workdir="./../data/", IP_dist=gamma( a=30, scale=20/30), init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
tst.CalculateRts()
tst.PlotPostRt( i=n-1, ax=ax, linestyle='--', color='green')
ax1.set_xlim((0,30))
ax1.grid(color='grey', linestyle='--', linewidth=0.5)
ax1.set_ylabel(r"Density")
ax1.set_xlabel("days")
ax1.set_title("")
fig1.tight_layout()
fig1.savefig("../paper/figs/Covid19_SerialTimeDist.png")
ax.set_xlim((0.5,2.5))
last_date = tst.init_date + timedelta(tst.m)
ax.set_xlabel(r"$R_t$, " + (last_date-timedelta(1)).strftime("%d.%m.%Y"))
ax.grid(color='grey', linestyle='--', linewidth=0.5)
ax.set_title("")
fig.tight_layout()
fig.savefig("../paper/figs/%s_Rts_Compare1.png" % (clave,))

trim=-0 ## Number of days to cut data from the end, negative, e.g. -10, cut 10 days
fig, ax = subplots( num=32, figsize=( 4.5, 3.5))
t0 = time()
for i in range(20):
    a = 3 + 4*uniform.rvs()-2
    mean = 8 + 14*uniform.rvs()-7
    scale = mean/a
    ### Plota the erlang( a=5, scale=9/5 ) alternative
    #PlotFrozenDist( gamma( a=a, scale=scale ), ax=ax1, color='grey', alpha=0.7)

    tst = Rts_AR( clave, workdir="./../data/", init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
    tst.CalculateRts()
    tst.PlotPostRt( i=n-1, ax=ax)
    #### Here we change the serial time:    Any other positive density could be used.
    tst = Rts_AR( clave, workdir="./../data/", IP_dist=gamma( a=a, scale=scale), init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
    tst.CalculateRts()
    tst.PlotPostRt( i=n-1, ax=ax, color='grey', alpha=0.7)
print("Time to calculate 20 posteriors: %f s." % (time()-t0,))
### Analysis with the assumed COVUD19 Serial interval
PlotFrozenDist( erlang( a=3, scale=8/3 ), ax=ax1)
tst = Rts_AR( clave, workdir="./../data/", init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
tst.CalculateRts()
tst.PlotPostRt( i=n-1, ax=ax)
tst = Rts_AR( clave, workdir="./../data/", IP_dist=gamma( a=3, scale=8/3), init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
tst.CalculateRts()
tst.PlotPostRt( i=n-1, ax=ax, color='black')
### Analysis with a very different Serial interval
PlotFrozenDist( erlang( a=30, scale=20/30 ), ax=ax1, linestyle='--', color='green')
tst = Rts_AR( clave, workdir="./../data/", init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
tst.CalculateRts()
tst.PlotPostRt( i=n-1, ax=ax)
tst = Rts_AR( clave, workdir="./../data/", IP_dist=gamma( a=30, scale=20/30), init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
tst.CalculateRts()
tst.PlotPostRt( i=n-1, ax=ax, linestyle='--', color='green')
ax.set_xlim((0.5,2.5))
last_date = tst.init_date + timedelta(tst.m)
ax.set_xlabel(r"$R_t$, " + (last_date-timedelta(1)).strftime("%d.%m.%Y"))
ax.grid(color='grey', linestyle='--', linewidth=0.5)
ax.set_title("")
fig.tight_layout()
fig.savefig("../paper/figs/%s_Rts_Compare2.png" % (clave,))




### Plot the Rt's estimation.  Only Merida, '13-01'  and Mexico city, '9-01', are in the paper
claves = ['15-02', '17-02', '23-01', '25-01', '12-01',  '9-01', "31-01"] 

x_jump = 7 ## For ploting, put ticks every x_jump days.

for i,clave in enumerate(claves):
    print(clave)
    ### Open an instance of the Rts_AR class: 
    tst = Rts_AR( clave, workdir="./../data/", init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
    tst.CalculateRts() # Most be called before ploting the Rt's
    ### Plot the Rts:
    fig, ax = subplots( num=i+1, figsize=( 8, 3.5))
    ### Plot Cori et al (2013) Poisson model version:
    PlotRts_P( '../data/%s.csv' % (clave,), init_date=ZMs[clave][3]+timedelta(days=4),\
            n=tst.n, trim=trim, ax=ax, color='green', alpha=0.5, median_color='black')
    ### Plot ours:
    tst.PlotRts( ax=ax, x_jump=x_jump, plot_area=[0.4,2.2], csv_fnam=clave, plot_obs_rts=True)
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_title("")
    ax.set_ylabel(r"$R_t$")
    ax.set_xlabel("")
    ax.set_title(ZMs[clave][0] + ", Mexico")
    fig.tight_layout()
    fig.savefig("../paper/figs/%s_Rts_AR.png" % (clave,))
    if clave == '9-01':
        m_max = tst.m
ax.set_xlabel("day.month, 2020")
fig.tight_layout()
fig.savefig("../paper/figs/%s_Rts_AR.png" % (clave,))

### Figure with Cori et al (2013) posterior distributions of '31-01' and '9-01'
fig1, ax1 = subplots( num=20, nrows=1, ncols=2, figsize=( 10, 3.5))
color = [ "red", "black", "darkred"]
for i,clave in enumerate([ '31-01', '9-01']):        
    tst = Rts_AR( clave, workdir="./../data/", init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
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
tst = Rts_AR( clave, workdir="./../data/", init_date=ZMs[clave][3]+timedelta(days=4), trim=trim, pred=0, n=n)
a, b = Rts_P( tst.data*10, tau=7, n=30, q=2)
ax1[0].plot( arange(m_max-tst.m, m_max, 1), tst.data*10, '.-', color=color[2])
PlotFrozenDist( gamma( a[-1], scale=b[-1]), ax=ax1[1], color=color[2])
ax1[1].set_xticks(arange(0.8,1.4,0.2))
ax1[1].set_xlabel(r"$R_t$, " + (last_date-timedelta(1)).strftime("%d.%m.%Y")) 
ax1[1].grid(color='grey', linestyle='--', linewidth=0.5)

fig1.tight_layout()
fig1.savefig("../paper/figs/Rts_Compare.png")


