#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 18:52:58 2021
@author: fabien
"""

import numpy as np, pylab as plt
import networkx as nx

import cv2
from skimage.feature import peak_local_max

################################ GRAPH of Network
class GRAPH():
    def __init__(self, I, O, PATTERN = None, SAVE_NETX = True):
        # assign parameter
        self.IO = I, O
        self.N = np.rint(np.mean([I,O])).astype(int)
        # Patterning : density of node repartition
        if PATTERN == None :
            # Sinusoidal patterning
            self.HPATTERN, self.LABEL_MASK = self.PATTERNNING(I,O)
        else : self.HPATTERN = PATTERN
        # nodes following distribution pattern (+labelling)
        self.NODE = self.RANDOM_GEO_NODE()
        # define radius
        self.RADIUS = self.R_CALC()
        # construct epured geometric graph
        self.GRAPH_ANN = self.EPURATE_GEO_GRAPH()
        # complete graph
        self.COMPLETE_GRAPH()
        # convert to neural form
        self.NEURON_LIST = self.CONVERTION2NN()
        # networkx
        if SAVE_NETX :
            # save graph
            self.G = self.NETX_GGRAPH()
            self.pos_gg = nx.get_node_attributes(self.G,'pos')
            self.color = nx.get_node_attributes(self.G,'color')
            # save nn
            self.H = self.NETX_ANN()
            self.pos_nn = nx.get_node_attributes(self.H,'pos')
            self.w_nn = list(nx.get_edge_attributes(self.H,'weight').values())
        
    def PATTERNNING(self, I,O):
        ### DRAW PATTERN
        SIDE = 10
        # GRADIENT DENSITY
        GRAD = (((O-I)/SIDE)*np.mgrid[0.1:3*np.pi:.1]+I)**2
        GRAD = GRAD/GRAD.sum() # normalisation
        # SINUS PATTERN
        self.X, self.Y = np.mgrid[0.:3*np.pi:.1,0.1:3*np.pi:.1]
        Z = (np.sin(self.X)*np.cos(np.pi/2+self.Y))**2
        # MASK
        Z_norm = cv2.normalize(Z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ret, Z_mask = cv2.threshold(Z_norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # PATTERN
        P = (GRAD*Z)*Z_mask
        E = (P/P.sum()).T
        ### LABELING (Watershed)
        dist_transform = cv2.distanceTransform(Z_mask,cv2.DIST_L2,5)
        local_maxi = peak_local_max(dist_transform, indices=False, labels=Z_mask)
        ret, markers = cv2.connectedComponents(np.uint8(local_maxi))
        markers = cv2.watershed(cv2.cvtColor(Z_mask,cv2.COLOR_GRAY2BGR),markers)
        MARKER = markers*Z_mask+1
        return E, MARKER
    
    def RANDOM_GEO_NODE(self):
        # generate node
        x, y = self.X.ravel(), self.Y.ravel()
        DENSITY = self.HPATTERN.ravel()
        node = np.random.choice(x.size, self.N, p=DENSITY)
        ## construct position array
        Input = np.concatenate((-0.5*np.ones(self.IO[0])[None],np.linspace(1,9,self.IO[0])[None])).T
        Output = np.concatenate((10*np.ones(self.IO[1])[None],np.linspace(1,9,self.IO[1])[None])).T
        Hidden = np.concatenate((x[None,node],y[None,node])).T
        xy = np.concatenate((Input,Hidden,Output)).T.astype(object)
        # labelling
        IN_LABEL, OUT_LABEL = np.zeros(self.IO[0]), -1*np.ones(self.IO[1])
        HIDDEN_LABEL = self.LABEL_MASK.ravel()[node]
        LABEL = np.concatenate((IN_LABEL,HIDDEN_LABEL,OUT_LABEL))[None].astype(np.int)
        # return nodes info
        return np.concatenate((np.arange(xy.shape[1], dtype=np.int)[None],xy,LABEL)).T
    
    def R_CALC(self):
        centroid = []
        # calculate centroid of node per label
        for m in np.unique(NET.NODE[:,3]) :
            bool_mark = NET.NODE[:,3] == m
            X,Y = NET.NODE[:,1:3].T
            centroid += [[np.mean(X[bool_mark]), np.mean(Y[bool_mark])]]
        # adding limit and conv2array
        centroid += [[-0.5,0],[-0.5,10], [10,0],[10,10]]
        centroid = np.array(centroid)
        # calculate nearness centroid for each, and find the max
        dist_min = []
        for c in centroid :
            dist_min += [np.linalg.norm(centroid[(centroid!=c).all(axis=1)]-c, axis=1).min()]
        return max(dist_min)
    
    def EPURATE_GEO_GRAPH(self):
        I, O = self.IO
        # define connection (1) or perceptron (-1)
        TYPE = np.zeros((self.N+I+O,1), dtype=int)
        TYPE[:I] = -1
        TYPE[I+self.N:] = 1
        # define probability connection
        PROBA = np.ones((self.N+I+O,1), dtype=float)
        # concat (use namedtuple ?)
        DATA = np.concatenate((self.NODE, TYPE, PROBA), axis=1)
        ### connect near neighbors and update type
        CONNECTED = np.array(len(DATA)*[[None]], dtype=object)
        for d in DATA[::-1]:
            # verifing if indef type or input
            if d[-2] != -1 :
                # calculate Visual field for each neuron
                P_XY = DATA[:, 1:3].astype(float)
                vertice = np.linalg.norm(P_XY-d[1:3].astype(float),axis=1) < self.RADIUS
                # extract data vertice
                data = DATA[vertice]
                # eliminate incoherent node
                data = data[data[:,-3] != d[-3]]
                data = data[data[:,-2] != 1]
                # calculate euclidean dist
                V = data[:,1:3].astype(float)
                if V.size != 0 :
                    dist = np.linalg.norm(V-d[1:3].astype(float),axis=1)
                    # artificial proba connection (Odds ratio and front/back)
                    dist = dist*data[:,-1]
                    sign = np.sign(d[1]-V[:,0]); sign[sign==-1] = np.pi
                    dist = sign*dist
                    # choose node
                    ndist = abs(dist-dist.max()).astype(float)
                    if dist.size == 1 or ndist.sum() == 0 : idx = 0
                    else : idx = np.random.choice(dist.size, p=ndist/ndist.sum()) # dist==dist.min()
                    data_min = data[idx]
                    CONNECTED[d[0]] = data_min[0]
                    # update DATA TYPE & Proba (+10%)
                    DATA[data_min[0],-2] = -1
                    DATA[data_min[0],-1] += .25
        return np.concatenate((DATA, CONNECTED), axis=1)
    
    def COMPLETE_GRAPH(self):
        # initialisation
        I, O = self.IO
        DATA, CONNECTED = self.GRAPH_ANN[:,:-1], self.GRAPH_ANN[:,-1]
        x, y = self.X.ravel(), self.Y.ravel()
        DENSITY = self.HPATTERN.ravel()
        MARKERS = self.LABEL_MASK.ravel()
        # new data buffer
        NEW_DATA, i = [], 0
        NEW_CONNECTED = []
        for d in DATA[:I]:
            if not (CONNECTED == d[0]).any() :
                new_node = np.random.choice(x.size, p=DENSITY)
                new_xy = [x[new_node],y[new_node]]
                new_m = MARKERS[new_node]
                NEW_DATA += [[I+self.N+i]+new_xy+[new_m]+ [1]+ [1.]]
                NEW_CONNECTED += [d[0]]
                i += 1
        NEW_DATA, NEW_CONNECTED = np.array(NEW_DATA), np.array(NEW_CONNECTED)
        ## reconstruct data & connect
        DATA_ = np.concatenate((DATA[:I+self.N],NEW_DATA,DATA[-O:]))
        CONNECT_ = np.concatenate((CONNECTED[:I+self.N], NEW_CONNECTED, CONNECTED[-O:]))
        DATA_[-O:,0] += i
        DATA_[:,[0,3]] = DATA_[:,[0,3]].astype(int)
        # update graph ann
        self.GRAPH_ANN = np.concatenate((DATA_, CONNECT_[:,None]), axis=1)
    
    def NETX_GGRAPH(self):
        G = nx.DiGraph()
        # get graph info
        pos = dict(enumerate(self.GRAPH_ANN[:,1:3].tolist()))
        color = dict(enumerate(np.char.add('C', (1+self.GRAPH_ANN[:,3]).astype(str))))
        # add node
        G.add_nodes_from(self.GRAPH_ANN[:,0])
        nx.set_node_attributes(G, pos, "pos")
        nx.set_node_attributes(G, color, "color")
        # add edges
        i2o = self.GRAPH_ANN[:,-1] != None
        G.add_edges_from(list(map(tuple,self.GRAPH_ANN[i2o][:,[-1,0]])))
        return G
    
    def CONVERTION2NN(self):
        # extract index label        
        IDX = np.unique(self.GRAPH_ANN[:,3])
        # idx, neuron, connect, x, y, link
        NEURON_LIST = np.zeros((IDX.size,6), dtype=object)
        NEURON_LIST[:,0] = IDX
        # merges label group
        for n in range(IDX.size):
            idx = IDX[n]
            # param
            data = self.GRAPH_ANN[self.GRAPH_ANN[:,3]==idx]
            NEURON_LIST[n,3] = np.mean(data[:,1]) # x
            NEURON_LIST[n,4]  = np.mean(data[:,2]) # y
            NEURON_LIST[n,1] = len(data[data[:,-1]==None]) # n
            NEURON_LIST[n,2] = len(data[data[:,-1]!=None]) # c
            # adding link neuron/connec
            list_c = []
            for dd in data[data[:,-1]!=None, -1] :
                layer = self.GRAPH_ANN[dd,3]
                neuron = np.where(self.GRAPH_ANN[self.GRAPH_ANN[:,3]==layer,0]==dd)[0][0]
                list_c += [[layer,neuron]]
            NEURON_LIST[n,-1] = list_c
        return NEURON_LIST
    
    def NETX_ANN(self):
        # reconstruct graph
        H = nx.DiGraph()
        # add node
        for nn in self.NEURON_LIST :
            H.add_node(nn[0],pos=nn[3:5])
        # add edges
        for nn in self.NEURON_LIST :
            if nn[2] != 0 :
                uni, ret = np.unique(np.array(nn[-1])[:,0], return_counts=True)
                for u,r in zip(uni,ret):
                    H.add_edge(u, nn[0], weight=r)
        return H
    
if __name__ == '__main__' :
    NET = GRAPH(100,10)
    plt.matshow(NET.LABEL_MASK); plt.show();plt.close()
    print(NET.GRAPH_ANN)
    # plot graph and network
    nx.draw_networkx_edges(NET.G, NET.pos_gg, alpha=0.1)
    nx.draw_networkx_nodes(NET.G, NET.pos_gg, node_color=list(NET.color.values()), node_size=10, cmap=plt.cm.Reds_r)
    plt.show();plt.close()
    nx.draw_networkx_edges(NET.H, NET.pos_nn, alpha=0.1, width=NET.w_nn)
    nx.draw_networkx_nodes(NET.H, NET.pos_nn, node_size=100, cmap=plt.cm.Reds_r)
    plt.show(); plt.close()
    
