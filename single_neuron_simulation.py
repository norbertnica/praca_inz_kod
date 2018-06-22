
# coding: utf-8

# In[1]:

# SKRYPT PREZENTUJE DYNAMIKĘ STANU WEWNĘTRZNEGO CHAOTYCZNEGO NEURONU
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
fig_size = [10, 8]
plt.rcParams["figure.figsize"] = fig_size


# In[2]:

#parametry dla dynamiki przypominajacej odwzorowanie logistyczne
e = 0.015
k = 0.5
alpha = 1
trans = 100

#parametry blizsze parametrom uzywanym w pracy
#k = 0.9
#alpha = 12


# In[3]:

def f(y):
    return 1/(1+np.e**(-y/e))

def g(x):
    return f(x)

def y(y_old,a): # rownanie na stan wewnetrzny neuronu
    return k*y_old - alpha*g(y_old) + a


# In[4]:

a = np.arange(-0.05,0.045,0.00001) 
it = 1100 
# dla drugiego zestawu parametrow uzywano:
# a = np.arange(0,1,0.00001)


# In[5]:

y_arr = np.zeros((len(a),it))
for n in np.arange(len(a)):
    temp = []
    temp.append(0)
    for i in range(it-1):
        temp.append(y(temp[i],a[n]))
    y_arr[n,:] = temp


# In[6]:

for i in range(it-trans):
    plt.scatter(a,y_arr[:,i+trans],s=1,color='k',alpha=0.3)


# In[ ]:



