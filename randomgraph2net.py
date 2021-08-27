#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:09:48 2021
@author: fabien
"""

import numpy as np, pylab as plt
import networkx as nx

from sklearn.cluster import Birch
from sklearn import preprocessing


## Global parameter
N = 300
I, O = 100, 100

## density gradient
a = (O-I)/10
g = a*np.mgrid[0.1:3*np.pi:.1]+I
E1 = (g/g.sum())

## construct density hidden grid
X, Y = np.mgrid[0.1:3*np.pi:.1,0.1:3*np.pi:.1]
Z = (np.sin(X)*np.cos(Y))**10
Z = E1*Z
E = (Z/Z.sum()).T # density
plt.contourf(X,Y,E); plt.show(); plt.close()

## random variable position
x, y = X.ravel(), Y.ravel()
node = np.random.choice(x.size, N, p=E.ravel())
plt.plot(x[node],y[node], 'o', ms=1); plt.show(); plt.close()

## construct position array
Input = np.concatenate((-0.5*np.ones(I)[None],np.linspace(1,9,I)[None])).T
Output = np.concatenate((10*np.ones(O)[None],np.linspace(1,9,O)[None])).T
Hidden = np.concatenate((x[None,node],y[None,node])).T

# convert to dict
xy = np.concatenate((Input,Hidden,Output))
pos = dict(enumerate(xy.tolist()))

## construct geometric graph
G = nx.random_geometric_graph(N+I+O, 2.5, pos=pos)

## clustering
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(Hidden)    

cluster = Birch(n_clusters = 32, threshold=X.std()/np.pi).fit(X)
label = cluster.labels_

# draw cluster with graph
fig = plt.figure(figsize=(8, 8))

nx.draw_networkx_edges(G, pos, alpha=0.1)
nx.draw_networkx_nodes(G, pos, node_size=20, cmap=plt.cm.Reds_r)

ax = fig.add_subplot(111)
for l in np.unique(label):
    ax.plot(Hidden[label==l,0],Hidden[label==l,1], 'o', ms=5)
plt.axis("off"); plt.show(); plt.close()
