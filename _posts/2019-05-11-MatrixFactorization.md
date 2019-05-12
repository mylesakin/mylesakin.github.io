---
title: "Matrix Completion"
date: 2019-05-11
tags: [Matrix Completion, Recommender System]
excerpt: "Matrix Completion for Recommender Systems"
mathjax: "true"

$$
\begin{bmatrix}
1 & 2 & 4 & 5 & 4 & 1\\
1 & 1 & 5 & 4 & 5 & 2\\
5 & 4 & 2 & 2 & 1 & 5\\
4 & 5 & 1 & 1 & 2 & 5
\end{bmatrix}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1,1,5,4],[2,1,4,5],[4,5,2,1],[5,4,2,1],[4,5,1,2],[1,2,5,5]])
print(A)
```

    [[1 1 5 4]
     [2 1 4 5]
     [4 5 2 1]
     [5 4 2 1]
     [4 5 1 2]
     [1 2 5 5]]



```python
A_norm = A-np.mean(A)
print(A_norm)
```

    [[-2. -2.  2.  1.]
     [-1. -2.  1.  2.]
     [ 1.  2. -1. -2.]
     [ 2.  1. -1. -2.]
     [ 1.  2. -2. -1.]
     [-2. -1.  2.  2.]]



```python
A_obs = np.array([[1,0,5,4],[0,1,4,5],[4,5,2,0],[5,0,2,1],[0,5,1,2],[1,2,0,5]])
A_o = A_obs.T#-np.mean(A_obs)
print(A_o)
```

    [[1 0 4 5 0 1]
     [0 1 5 0 5 2]
     [5 4 2 2 1 0]
     [4 5 0 1 2 5]]



```python
q = np.random.uniform(size=[3,6]) #items
p = np.random.uniform(size=[3,4]) #users
print(p)
print(q)
```

    [[0.62972638 0.47657213 0.43784504 0.46812787]
     [0.80342236 0.81407323 0.03337737 0.25106506]
     [0.88671324 0.55631268 0.32038436 0.85217001]]
    [[0.24320148 0.45891692 0.80990547 0.44920768 0.68263868 0.82094832]
     [0.60291308 0.20622186 0.01389857 0.43044343 0.35589774 0.09460499]
     [0.93032012 0.65807887 0.44992909 0.92782132 0.76945596 0.66117582]]



```python
q = np.random.rand(2,6) #items
p = np.random.rand(2,4) #users
a,b = A_o.shape
for i in range(10000):
    for j in range(a):
        for k in range(b):
            if A_o[j,k]==0:
                continue
            else:
                #print(A_obs.T[j,k])
                e = A_o[j,k]-np.dot(p[:,j].T,q[:,k])
                #print(e)
                q[:,k] = q[:,k]+0.0002*(2*e*p[:,j]-0.02*q[:,k])
                #print(q)
                p[:,j] = p[:,j]+0.0002*(2*e*q[:,k]-0.02*p[:,j])


X = np.dot(p.T,q)
print(X)#+np.mean(A_obs))
print(A_o)#+np.mean(A_obs))
print(A.T)
print(np.mean(np.multiply(X-A.T,X-A.T)))
```

    [[1.02956467 0.1601237  3.97832855 4.98332674 3.99881428 0.9786323 ]
     [1.99846817 0.98344889 4.99527503 6.09803306 4.94875368 2.01771554]
     [4.44239443 4.43328222 1.99356456 1.61176413 1.60188457 4.87977101]
     [4.51077194 4.5508638  1.82419916 1.37431126 1.42016051 4.96354523]]
    [[1 0 4 5 0 1]
     [0 1 5 0 5 2]
     [5 4 2 2 1 0]
     [4 5 0 1 2 5]]
    [[1 2 4 5 4 1]
     [1 1 5 4 5 2]
     [5 4 2 2 1 5]
     [4 5 1 1 2 5]]
    0.4764523099119438



```python
np.random.rand(2)
```




    array([0.50496673, 0.3044647 ])




```python

X=A_o
t=5
for i in range(5000):
    u,ep,v = np.linalg.svd(X)
    c = np.where(ep>t)[0]
    uu = u[:,c]
    vv = v[c,:]
    ec = ep[ep>t]
    ec = np.multiply(np.identity(len(c)),ec)
    D = np.copy(A_o)
    D[D>0]=1
    X_k = np.dot(u[:,c],np.dot(ec,v[c,:]))
    X = X_k-0.002*(np.multiply(X_k,D)-A_o)

print(X)
print(A_o)
print(A.T)
print(np.mean(np.multiply(X-A.T,X-A.T)))
```

    [[0.90131017 0.43381772 4.11780359 4.87206391 4.2284632  1.13236092]
     [1.56493354 1.03796054 4.89637006 5.69426296 4.96866177 1.88181208]
     [4.45518519 4.46108367 1.98934987 1.55559806 1.56233731 4.90289122]
     [4.52049666 4.54653898 1.89221998 1.42135115 1.45426918 4.97237035]]
    [[1 0 4 5 0 1]
     [0 1 5 0 5 2]
     [5 4 2 2 1 0]
     [4 5 0 1 2 5]]
    [[1 2 4 5 4 1]
     [1 1 5 4 5 2]
     [5 4 2 2 1 5]
     [4 5 1 1 2 5]]
    0.3567015824625475
