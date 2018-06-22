import numpy as np
import math
import copy
import random

def network_step(x_old, eta_old, zeta_old, w, a): # funkcja odtwarzajaca jeden krok sieci, w to macierz wag, a to stymulacja
    #parametry dla ktorych przeprowadzano wszystkie symulacje
    k_f = 0.8
    k_r = 0.9
    alpha = 12
    e = 0.015
    
    def f(y): #funkcja aktywacji neuronu
        return 1/(1+np.e**(-y/e))
    
    def f_eta(eta_old): 
        return k_f * eta_old + np.dot(x_old,w)
    
    def f_zeta(zeta_old):
        return k_r * zeta_old - alpha * x_old + a
    
    eta = f_eta(eta_old)
    zeta = f_zeta(zeta_old)
    x = f(eta + zeta)
    return x, eta, zeta

def hebb(pat): #implementacja reguly Hebba
    p = np.shape(pat)[0]
    N = np.shape(pat)[1]
    w = np.zeros((N,N))
    for i in range(p):
        w = w + np.outer(pat[i,:],pat[i,:])
    w = w/p
    return w

def gram_schmidt(X): #implementacja algorytmu Grama-Schmidta
    def proj(u,v):
        return (np.dot(u,v)/np.dot(u,u))*u
    M = np.zeros((np.size(X,0),np.size(X,1)))
    M_nor = np.zeros((np.size(X,0),np.size(X,1)))
    for i in range(np.size(X,0)):
        M[i,:] = X[i,:].copy()
        for j in range(i):
            M[i,:] = M[i,:] - proj(M[j,:],X[i,:])  
        M_nor[i,:] = M[i,:]/np.linalg.norm(M[i,:])    
    return M, M_nor #zwraca w ogolnosci dwa elementy, M to macierz ortogonalnych wektorow, M_nor wektorow po normaliacji

def overlap(x,pat,N): #funkcja liczaca przekrycia stanu sieci x z kazdym ze wzorcow z macierzy wzorcow pat
    p = np.shape(pat)[0]
    mu = np.zeros(p)
    for i in range(p):
        mu[i] = np.dot(2*x-1,pat[i,:])/N
    return mu
    