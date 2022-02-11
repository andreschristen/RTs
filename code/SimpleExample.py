#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:33:53 2022

@author: jac
"""


from datetime import date

from numpy import array, floor

from Rts_AR import Rts_AR



### Data for Mérida city contained in 
### ./../data/31-01.csv
### This is a two column csv file, column 0 with deaths and
### column 1 with reported cases.  We use reported cases as incidence data.

init_date = date(2020, 3,  11) #Date of first row (dayly reports are assumed)

### The reported cases are copied here:
data =\
array([  1.,   2.,   1.,   3.,   3.,   1.,   5.,   3.,   1.,   2.,   7.,\
         5.,   9.,   3.,   4.,   4.,   3.,   3.,   1.,   2.,   4.,   5.,\
         5.,   5.,   4.,   7.,   6.,   9.,   6.,   4.,  10.,  11.,  11.,\
         3.,   9.,  17.,  15.,  14.,   5.,  10.,  11.,  13.,  17.,  21.,\
        19.,  25.,   9.,  22.,  23.,  24.,  27.,  33.,  22.,  13.,  26.,\
        30.,  24.,  16.,  27.,  21.,  23.,  30.,  32.,  33.,  36.,  38.,\
        26.,  30.,  28.,  44.,  37.,  37.,  39.,  21.,  34.,  42.,  38.,\
        39.,  46.,  38.,  35.,  27.,  36.,  37.,  32.,  25.,  43.,  34.,\
        33.,  43.,  33.,  47.,  41.,  59.,  56.,  64.,  46.,  57.,  59.,\
        53.,  86.,  80.,  55., 109.,  84., 129.,  78., 110.,  76.,  91.,\
       103., 105., 102.,  92., 100.])

### Here we open an instance of the Rts_AR class:
### In this case, we are using the above array with the incidence data
### We ask for n=30 Rts, from the last date through n=30 days before.
n=30  
tst = Rts_AR( data, init_date=init_date, n=n)

### Reading from the file would be like this
#tst = Rts_AR( data_fnam="31-01", workdir="./../data/", col=1, init_date=init_date, n=n)

###  Please see the rest of parameters for the class Rts_AR with
#help(Rts_AR)

### Here we calculate all the n=30 Rts
tst.CalculateRts()

### Plot the last one.  If axes ax=None, PlotPostRt creates an axes
### and returns it:
ax = tst.PlotPostRt( i=n-1, ax=None)
ax.set_xlim((0.5,2))
ax.set_title("Mérida")
fig = ax.get_figure()
fig.tight_layout()
#fig.savefig("Merida.png")
### This last density ios stored in tst.rt and tst.pdf
### To calculate, for example, the probability that Rt is below a=1
a=1
### integrate with
DeltaRt = tst.rt[1]-tst.rt[0] #A uniform grid is always used, with linspace
i = int(floor((a-tst.rt[0])/DeltaRt))
prob = sum( tst.pdf[:(i+1)]*DeltaRt ) + tst.pdf[i+1]*(a-tst.rt[i])
print("Mérida, $P[R_t < %f | Y] = %6.4f$" % (a,prob))

### Make a plot with quantile plots of all calculated Rts posteriors
ax = tst.PlotRts()
ax.set_title(r"Mérida $R_t$s")
fig = ax.get_figure()
fig.tight_layout()


