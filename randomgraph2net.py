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

import cv2
from skimage.feature import peak_local_max

## Global parameter
N = 300
I, O = 100, 50

########################################## CONSTRUCT GRAPH
## density gradient
a = (O-I)/10
g = a*np.mgrid[0.1:3*np.pi:.1]+I
E1 = (g/g.sum())

## construct density hidden grid
X, Y = np.mgrid[0.1:3*np.pi:.1,0.1:3*np.pi:.1]
Z = (np.sin(X)*np.cos(np.pi/2+Y))**10

Z_norm = cv2.normalize(Z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
ret, Z_mask = cv2.threshold(Z_norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

Z1 = (E1*Z)*Z_mask
E = (Z1/Z1.sum()).T # density
plt.contourf(X,Y,E); plt.show(); plt.close()

## segmentation
# Watershed 
dist_transform = cv2.distanceTransform(Z_mask,cv2.DIST_L2,5)
local_maxi = peak_local_max(dist_transform, indices=False, labels=Z_mask)
ret, markers = cv2.connectedComponents(np.uint8(local_maxi))
markers = cv2.watershed(cv2.cvtColor(Z_mask,cv2.COLOR_GRAY2BGR),markers)
markers = markers*Z_mask
plt.imshow(markers); plt.show(); plt.close()
# localisation
loc = []
for m in np.unique(markers) :
    loc += [[np.mean(X[markers == m]), np.mean(Y[markers == m])]]
loc = np.array(loc)
dist_min = []
for l in loc :
    dist_min += [np.linalg.norm(loc[(loc!=l).all(axis=1)]-l, axis=1).min()]
RADIUS = max(dist_min)
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
G = nx.random_geometric_graph(N+I+O, RADIUS, pos=pos)

## clustering
"""
SEGMENTATION PLUTOT
"""

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

"""
A = nx.convert_matrix.to_numpy_array(G)
plt.imshow(A);plt.show();plt.close()
"""
########################################## EPURATION GRAPH (conserv recurance ?)
### initialisation
# collect basic info
IDX = np.array(list(pos.keys()), dtype=int)
XY = np.array(list(pos.values()), dtype=object)
LABEL = np.concatenate((np.zeros(I), label+1, -1*np.ones(O))).astype(int)
# define connection (1) or perceptron (-1)
TYPE = np.zeros((N+I+O,1), dtype=int)
TYPE[:I] = -1
TYPE[I+N:] = 1
# define probability connection
PROBA = np.ones((N+I+O,1), dtype=float)
# concat (use namedtuple ?)
DATA = np.concatenate((IDX[:,None],XY,LABEL[:,None], TYPE, PROBA), axis=1)
### connect near neighbors and update type
CONNECTED = np.array(len(DATA)*[[None]], dtype=object)
for d in DATA[::-1]:
    # verifing if indef type or input
    if d[-2] != -1 :
        # extract link
        vertice = np.fromiter(G.neighbors(d[0]), dtype=int)
        # extract data vertice
        data = DATA[vertice]
        # eliminate incoherent node
        data = data[data[:,-3] != d[-3]]
        data = data[data[:,-2] != 1]
        # calculate euclidean dist
        X = data[:,1:3].astype(float)
        if X.size != 0 :
            dist = np.linalg.norm(X-d[1:3].astype(float),axis=1)
            # artificial proba connection (Odds ratio)
            dist = dist*data[:,-1]
            # choose node
            data_min = data[dist==dist.min()][0]
            CONNECTED[d[0]] = data_min[0]
            # update DATA TYPE & Proba (+10%)
            DATA[data_min[0],-2] = -1
            DATA[data_min[0],-1] += .1

# reconstruct graph
H = nx.Graph()
# add node
for d in DATA :
    H.add_node(d[0],pos=d[1:3])
# add edges
for n in range(len(CONNECTED)) :
    if CONNECTED[n] != None :
        H.add_edge(n,CONNECTED[n][0])

nx.draw_networkx_edges(H, pos, alpha=0.1)
nx.draw_networkx_nodes(H, pos, node_size=20, cmap=plt.cm.Reds_r)

"""
Il y a trois probleme :
    - Les premieres couches n'ont que une seule sortie...
        Ajouter probabilité de se connecter si est deja attribué comme sortie ?
    - La seconde est que toutes les sortie des "entrée du reseau" ne sont pas connecté
        Ajouter des connections avec couches superieurs ? (de maniere artificiel) 
    - La derniere, si une couche est connecté à toute les noeuds d'une autre couche, il n'y a plus d'entrée..
        Ajouter connection artificiellement ? (proportionnel à nb de sortie ? ou au moins une seule ?)
"""






