# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plotCurves = True
plotTree = True

#Nelson-Siegel-Svensson Curve
def fwd(t, params):
    beta0, beta1, beta2, tau1, beta3, tau2 = params
    return beta0*np.ones(t.shape) + beta1*np.exp(-t/tau1) + beta2*t/tau1*np.exp(-t/tau1) + beta3*t/tau2*np.exp(-t/tau2)

def spot(t, params):
    beta0, beta1, beta2, tau1, beta3, tau2 = params
    return beta0*np.ones(t.shape) + beta1*(1-np.exp(-t/tau1))*(tau1/t) \
     + beta2*((1-np.exp(-t/tau1))*(tau1/t) - np.exp(-t/tau1)) \
     + beta3*((1-np.exp(-t/tau2))*(tau2/t) - np.exp(-t/tau2))
#######################################################################

params = (0.05, -0.03, -0., 1.2, 0, 2)
numSteps = 12
maturity = 1
t = np.linspace(maturity/numSteps, maturity, numSteps+1)
r = spot(t, params)
df = np.exp(- r * t)

#Plot Spot Curve & DF
if plotCurves:
    fig, ax = plt.subplots(2,1,figsize=(6,6))
    ax[0].plot(t,r)
    ax[0].set_title("Spot Rate")
    ax[1].plot(t,df)
    ax[1].set_title("Discount Factor")
    fig.show()

#Tree Plot Function
def showTree(t, tree):
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    for i in range(len(t)):
        for j in range(i+1):
            ax.plot(t[i], tree[i][j], '.b')
            if i<len(t)-1:
                ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j]], '-b')
                ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j+1]], '-b')
    fig.show()

#Get Drift Adjustment
def getDrift(df, tree, Q, dt):
    df_unadjusted = (np.exp(-tree[-1]*dt) * Q[-1]).sum()
    return (-1/dt)*np.log(df / df_unadjusted)

#Tree Building
sigma = 0.01
dt = t[1]-t[0]
p = 0.5

r0 = r[0]
tree = [np.array([r0])]
Q = [np.array([1])]
m = [0]
for i in range(1, len(t)):
    Qi = np.zeros(i+1)
    for j in range(i):
        Qi[j] += p * Q[i-1][j] * np.exp(-tree[i-1][j]*dt)
        Qi[j+1] += (1-p) * Q[i-1][j] * np.exp(-tree[i-1][j]*dt)
    Q.append(Qi)
    tree.append(np.linspace(r0+i*sigma*np.sqrt(dt), r0-i*sigma*np.sqrt(dt), i+1))
    drift = getDrift(df[i], tree, Q, dt)
    m.append(drift)
    tree[-1] += drift

if plotTree:
    showTree(t,tree)