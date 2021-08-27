#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:09:48 2021

@author: fabien
"""

import numpy as np, pylab as plt
import networkx as nx

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

## draw network
plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.1)
nx.draw_networkx_nodes(G, pos, node_size=20, cmap=plt.cm.Reds_r)
#plt.savefig("OUT/graph_nn.png", dpi=360)
plt.axis("off"); plt.show()


"""
# Use seed when creating the graph for reproducibility
G = nx.random_geometric_graph(500, 0.125)
# position is stored as node attribute data for random_geometric_graph
pos = nx.get_node_attributes(G, "pos")

# find node near center (0.5,0.5)
dmin = 1
ncenter = 0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if d < dmin:
        ncenter = n
        dmin = d

# color by path length from node near center
p = dict(nx.single_source_shortest_path_length(G, ncenter))

plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(p.keys()),
    node_size=80,
    node_color=list(p.values()),
    cmap=plt.cm.Reds_r,
)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis("off")
plt.show()
"""