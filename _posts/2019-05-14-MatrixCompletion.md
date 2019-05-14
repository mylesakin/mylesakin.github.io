---
title: "Matrix Completion"
date: 2019-05-11
tags: [Matrix Completion, Recommender System]
excerpt: "Matrix Completion for Recommender Systems: A detailed look at matrix completion methods for recommendation systems. This includes Singular Value Decompostion and Stochastic Gradient Descent factorization."
mathjax: "true"
---
## Matrix Approximation by Singular Value Decomposition

Electronic retailers have inundated consumers with a huge number of products to choose from. Therefore, matching consumers with the most appropriate product is important for consumer satisfaction and helping to maintain loyalty. To address the task of recommending the best product, many strategies have been developed. A review of most of these methods can be found in the wonderful resource *Recommendation Systems Handbook* edited by Ricci, et al (refered to here a RSH). For this post, I will be focusing on a particular methods, and a few associated algorithms, know as matrix completion.

Matrix completion is a collaborative filtering technique for finding the best products to recommend. To start, *collaborative filtering* is a strategy of recommendation based on similarity of past consumption patterns by users. For example, suppose user A and user B have rated movies in a similar manner in the past. We might then expect for a movie A has seen but B has not, and A has rated highly, B would also enjoy this movie. By this method, we can find products for users based on similar users. Another strategy I will not discuss more here is content filtering. *Content filtering* observes the similarity of product features in order to recommend new products. For instance, sticking with the movie theme, if a user enjoyed the movie Aliens, we might recommend Predator as they are action packed, sci-fi movies. For more on content filtering see RSH.

One of the most successful techniques for collaborative filtering has been matrix completion, also sometimes called matrix factorization (for further tehcniques, again see RSH). This methods generally relies on one of two types of data, explicit feedback of implicit feedback. Explicit feedback, such as the old NetFlix star rating system, provides information explicitly given about products by users/consumers. Implicit feedback is not directly provided by users or consumers, such as purchase history. For the purpose of this post, I will assume explicit feedback.

We start by constructing a ratings table where the rows are the users (consumers), the columns are the products and entries are the explicit ratings given by the user for a product. As matrix completion was made popular by the NetFlix competition, I will use mives as the product to recommend and a rating from 1 to five.

|         | Aliens | Predator | Pretty Woman | Sleepless in Seattle | Notting Hill | Terminator |
|---------|--------|----------|--------------|----------------------|--------------|------------|
| **Charles** | 1      | 2        | 4            | 5                    | 4            | 1          |
| **Laura**   | 1      | 1        | 5            | 4                    | 5            | 2          |
| **Mark**    | 5      | 4        | 2            | 2                    | 1            | 5          |
| **Simone**  | 4      | 5        | 1            | 1                    | 2            | 5          |

This table is complete and as we can see, has some clear patterns. Two users, Mark and Simone, clearly prefer sci-fi movies to romantic movies, while Charles and Laura are the opposite. Of course, this is fairly unrealistic, but is used to make a point. The main assumption of matrix completion is that what influences users to rate products high or low is based on only a few item features known as *latent features*. These latent features, which we generally cannot interpret, can be found by a low rank approximation of the rating matrix. The rank of the approximation matrix determines the number of latent features. The rating matrix is simply the entries of the ratings table:

$$M = \begin{bmatrix}
1 & 2 & 4 & 5 & 4 & 1\\
1 & 1 & 5 & 4 & 5 & 2\\
5 & 4 & 2 & 2 & 1 & 5\\
4 & 5 & 1 & 1 & 2 & 5
\end{bmatrix}$$

The rank of a rectangular matrix can be found by the number of non-zero singular values in *singular value decomposition*. Singular value decomposition decomposes a \\(m\times n\\) matrix \\(M\\) as follows:

$$ M = USV^T $$

Where, \\(U\\) and \\(V\\) are unitary matrices with sizes \\(m\times m\\) and \\(n\times n\\) respctively. The columns of these matrices are known as left and right singular vectors. The matrix \\(S\\) is a \\(m\times n\\) diagonal matrix with the singular values on the diagonal. For our example, we can factorize using the following python code:


```python
import numpy as np

M = np.array([[1,2,4,5,4,1],[1,1,5,4,5,2],[5,4,2,2,1,5],[4,5,1,1,2,5]])
U,S,V = np.linalg.svd(M)
```

This gives the following result (rounded to two decimal places)

$$U = \begin{bmatrix}
-0.47 & 0.49 & -0.47 & -0.56\\
-0.49 & 0.54 & 0.47 & 0.49\\
-0.53 & -0.48 & -0.53 & 0.48\\
-0.51 & -0.50 & 0.52 & -0.47
\end{bmatrix},
$$
$$
S = \begin{bmatrix}
 14.73& 0 & 0 & 0 & 0 & 0\\
 0 & 7.77 & 0 & 0 & 0 & 0\\
 0 & 0 & 1.58 & 0 & 0 & 0\\
 0 & 0 & 0 & 1.49 & 0 & 0
\end{bmatrix}
$$
$$
V^T = \begin{bmatrix}
-0.38 & -0.41 & -0.40 & -0.40 & -0.40 & -0.45\\
-0.42 & -0.36 & 0.42 & 0.41 & 0.41 & -0.42\\
-0.34 & 0.02 & -0.03 & -0.63 & 0.63 & 0.29\\
0.31 & -0.70 & 0.46 & -0.24 & -0.18 & 0.33\\
0.61 & -0.30 & -0.45 & 0.03 & 0.49 & -0.30\\
-0.29 & -0.35 & -0.49 & 0.46 & 0.02 & 0.58
\end{bmatrix}
$$

We see that two of the singular values are much larger than the other two, indicating the importance of these dimensions in the reconstruction. We therfore might be interested in how close our an approximation of the matrix \\(M\\) would be if we used only those two singular values and their corresponding singular vectors. This is called truncated singular value decomposition. The following code gives an approximation with the first two singular values.


```python
S_2 = np.diag(S[:2])
M_2 = np.around(np.dot(U[:,:2], np.dot(S_2,V[:2,:])),2)
E = M-M_approx
Error = (1/24)*np.sum(E**2)
```

$$U_{2} = \begin{bmatrix}
-0.47 & 0.49 \\
-0.49 & 0.54 \\
-0.53 & -0.48 \\
-0.51 & -0.50
\end{bmatrix},
$$

$$
S_2 = \begin{bmatrix}
 14.73& 0 \\
 0 & 7.77
\end{bmatrix}
$$

$$
V^T_2 = \begin{bmatrix}
-0.38 & -0.41 & -0.40 & -0.40 & -0.40 & -0.45\\
-0.42 & -0.36 & 0.42 & 0.41 & 0.41 & -0.42
\end{bmatrix}
$$

$$M_{2} = U_2S_2V_2^T= \begin{bmatrix}
1.01 & 1.44 & 4.36 & 4.32 & 4.33 & 1.50\\
1.03 & 1.49 & 4.69 & 4.65 & 4.65 & 1.54\\
4.49 & 4.52 & 1.65 & 1.65 & 1.65 & 5.01\\
4.50 & 4.50 & 1.34 & 1.36 & 1.36 & 4.99
\end{bmatrix}$$

As we can see, this is a fairly good approximation. Using mean square error given by

$$ MSE = \frac{1}{mn}\sum_{i}\sum_{j}((M)_{ij}-(M_{2})_{ij}))^2 = \frac{1}{mn}\|M - M_{2}\|^2_F$$

we get an error of 0.197. Note, \\( \| \|_F \\) is the Frobenius norm. Not bad. In fact, with respect to the Frobenius norm, this is the best approximation we can get with a rank 2 matrix. This follows from the Eckart-Young theorem:

**Theorem**: (Eckart and Young) Let \\(A = USV^T\\) with \\(rank(r)\\). Then for some \\(k\\) such that \\(0< k \leq r\\) and \\(A_k = U_kS_kV_k^T\\) is the \\(k\\) truncated SVD,

$$\|A-A_k\|_F = \min_{rank(B)\leq k}\|A-B\|_F$$

That is, the \\(k\\) truncated SVD of some matrix \\(A\\) is the best \\(k\\) rank approximation of \\(A\\).

So far, we have considered matrices in which all entries are known. However, this is generally not the case for recommendations systems. Users almost never review all possible product. Therefore, we need to be able to approximate the matrix knowing only a some of the values. This is where SVD comes in handy for matrix completion, by assuming user ratings are based on a lower dimensional set of features. Let's look at a version of the movie ratings matrix with a few entries removed (unobserved).

$$M = \begin{bmatrix}
1 & - & 4 & 5 & - & 1\\
- & 1 & 5 & - & 5 & 2\\
5 & 4 & 2 & 2 & 1 & -\\
4 & 5 & - & 1 & 2 & 5
\end{bmatrix}$$

In our python code, we will use zeros in place of unobserved entries.


```python
M_obs = np.array([[1,0,4,5,0,1],[0,1,5,0,5,2],[5,4,2,2,1,0],[4,5,0,1,2,5]])
```

One thing to be concerned about is how many observations do we need in order to be able to approximate it with a lower rank matrix? A straightforward requirement is that we need an observation for each row and column, that is for each user and movie in our example. There are stronger requirments, but that is beyond the scope of this post, see the paper *Exact Matrix Completion via Convex Optimization* by Candes and Recht. Suffice it say that our matrix meets the necessary requirements.

There are a many algorithms for matrix completion based on singular value decomposition, here I will first talk about one that makes explicit use of SVD, the singular value thresholding method, then I will discuss the more popular stochastic gradient descent (SGD) technique. I will also relate the SGD and AM techniques back to SVD.


## Matrix Completion with Singular Value Thresholding

The Singular Value Thresholding (SVT) I will discuss here was introduced in *A Singular Value Thresholding Algorithm for Matrix Completion* by Cai, Candes and Shen. Suppose we are given a matrix \\(M\\) consisting a set of observations \\(\Omega = (i,j)\\), that is \\(M_{ij}\\) is observed when \\((i,j)\in \Omega\\). We define the orthogonal projector \\(P_\Omega\\) on a matrix \\(X\\) to be

$$
P_\Omega(A) = \begin{cases}
 A_{ij}& \text{ if } (i,j)\in \Omega \\
 0 & \text{ otherwise }  
\end{cases}
$$

This will allow us to find the error between the observed entries of \\(M\\) and our low-rank approximation matrix. Our goal now is to find a low-rank approximation, \\(X\\), of the matrix \\(M\\) given only a few observations of it's entries. In other word, we want to minimize the rank of our matrix with the constraint that the observed values set \\(\\Omega\\) are equivalent in both \\(X\\) and \\(M\\).

$$ \text{ minimize } \|X\|_*\\
\text{ subject to } P_\Omega(X) = P_\Omega(M)
$$

Where \\(\|X\|_* = \sum_i \sigma_i(X)\\) is the called the *nuclear norm* and \\(\sigma_i(X)\\) are the singular values of \\(X\\). Techinically this is a convex relaxtion to the non-convex rank minimization problem, for more on that see the references paapers. Minimizing the nuclear norm will send some of the less important singular values to zero. This requires advanced semidefinite programming optimization methods that have problems with larger matrices due to solving huge systems of linear equations. However, we can simplify this by approximately solving the nuclear norm minimization probelm using singular value thresholding.

Let's restate our optimization problem as follows using the Frobenius norm to ensure the equivalence of the observes set \\(\Omega\\) in \\(X\\) and \\(M\\)

$$ \min_{X\in \mathbb{R}^{m\times n}} \frac{1}{2}\|P_\Omega(M-X)\|_F^2-\tau\|X\|_* $$

Let's also define the soft-thresholding operator as

$$
D_\tau(A) = UT_\tau(S)V^T \\
T_\tau(S)_{ii} = \begin{cases}
 S_{ij} - \tau& \text{ if } S_{ij} > \tau \\
 0 & \text{ otherwise }  
\end{cases}
$$

It was proved in the paper *A Singular Value Thresholding Algorithm for Matrix Completion* by Cai, Candes and Shen that the matrix \\(D_\tau(X)\\) is a solution to the restated minimization problem. The following algorithm finds the optimal matrix iteratively

1. Initialize \\(X^{(k)}\\) for \\(k=0\\) as \\(X^{(0)} = M\\) and set a threshold \\(\tau\\)
2. Apply the soft threshold \\(Y^{(k)} = D_\tau(X^{(k)})\\)
3. Set \\(X^{(k+1)} = X^{(k)}+\delta P_\Omega(Y^{(k)}-M)\\), \\(\delta\\) is the learning rate

Note: the learning rate \\(\delta\\) is usually a function of \\(k\\), decreasing as the number of steps in creases. Here I will leave \\(\delta\\) static for simplicity.

The following python code implements this algorithm and applies it to the sample movie ratings matrix.


```python
def svd_comp(A,t,n):
    X=A
    D = np.copy(A)
    D[D>0]=1
    for i in range(n):
        u,ep,v = np.linalg.svd(X)
        c = np.where(ep>t)[0]
        uu = u[:,c]
        vv = v[c,:]
        ec = ep[ep>t]-t
        X_k = np.dot(u[:,c],np.dot(np.diag(ec),v[c,:]))
        X = X_k-0.02*(np.multiply(X_k,D)-A)

    return X

X = svd_comp(M_obs,0.003,10000)

```

The resulting optimized matrix and the orginal complete matrix are

$$X = \begin{bmatrix}
1.02 & \color{red}{0.44} & 3.97 & 4.86 & \color{red}{3.19} & 0.99\\
\color{red}{0.64} & 1.02 & 4.91 & \color{red}{3.87} & 4.88 & 1.99\\
4.86 & 3.98 & 1.97 & 1.99 & 1.01 & \color{red}{3.62}\\
4.01 & 4.90 & \color{red}{2.24} & 1.00 & 1.99 & 4.89
\end{bmatrix}
$$
,$$M = \begin{bmatrix}
1 & \color{red}{2} & 4 & 5 & \color{red}{4} & 1\\
\color{red}{1} & 1 & 5 & \color{red}{4} & 5 & 2\\
5 & 4 & 2 & 2 & 1 & \color{red}{5}\\
4 & 5 & 1 & \color{red}{1} & 2 & 5
\end{bmatrix}$$

the missing entries, \\(\Omega^c\\) are colored red. As we can see, the algorithm does a decent job of completing the matrix. Overall, the MSE \\(X\\) is 0.2817; not too bad. Fine tuning the hyperparameters \\(\delta\\) and \\(\tau\\) through cross calidation, as well as letting both be functions of \\(k\\) should result in a better approximation.

Now based, on these results we can build a decision rule to recommend new movies, say if the filled in value is greater than 3. In this case, we would want to recommend Notting Hill to Charles, but not Predator. Thus this simple algorithm allows us to personalize recommendations for each user.

While this method is simple, it has some drawbacks. At each iteration, we must calculate the SVD of the \\(X^{(k)}\\) matrix. For large matrices, this can be prohibitively time consuming. We may thus search for a different method to find a low rank approximation.

## Stochastic Gradient Descent

Since we want to move away from calculating SVDs, we may need to reformulate the optimization problem. The most popular method for low rank approximation reformulates the approximate factorization of the matrix \\(M \approx WH^T\\) where \\(M\\) is still and \\(m\times n\\) matrix and now \\(W\\) and \\(H\\) are \\(m\times r\\) and \\(n\times r\\) matrices respectively. \\(r\\) is now the target rank of the approximation. So unlike the soft-thresholding method, in this case we must specify *a priori* the target rank for our approximation.

Using this new factorization, we can restate our optimization problem as

$$
\text{ minimize } \frac{1}{2}\|M-WH^T\|_F^2 - \lambda(\|W\|_F^2 +\|H\|_F^2)
$$

Here we use \\(L^2\\) regularization constraints on the matrices \\(W\\) and \\(H\\) to prevent overfitting. You may be asking some questions now; Where does this factorization come from? Why do we use this factorization when we know truncated SVD is the best solution?

To answer the first, consider the following truncated SVD for some matrix \\(A\\)

$$ A = U_rS_rV_r^T = (U_rS_r^{1/2})(S_r^{1/2}V_r^T) = WH^T\\
W = U_rS_r^{1/2}, H = V_rS_r^{1/2} $$

As we can see, the rank \\(r\\) solution \\(WH^T\\) can arise from truncated SVD. In fact, this is the optimal solution. What is interesting, is that right multiplication of \\(W\\) and \\(H\\) by an orthogonal \\(r\times r\\) matrix leaves the product \\(WH^T\\) unchanged! We will see the importance of this later.

As to the second question, while this is a non-convex optimization problem, we can still use stochastic gradient descent (SGD) to find an at least locally optimal solution. As you may know, SGD is incredibly simple to implement and is quick to run even on large matrices. The SGD algorithm for this particular problem is implemented as:

1. Initialize \\(W\\) and \\(H\\) to random values, typically uniformly over \\([0,1]\\)
2. For each observation \\((i,j) \in \Omega\\), calculate \\(e_{ij} = M_{ij} - W_iH^T_j\\), where \\(W_i\\) and \\(H_j\\) are the ith and jth columns respectively
3. Update matrices \\(W\\) and \\(H\\) with learning rate \\(\alpha\\):
    * \\(W_i \leftarrow W_i + \alpha(e_{ij}H_i-\lambda W_j)\\)
    * \\(H_i \leftarrow H_j + \alpha(e_{ij}W_j-\lambda H_i)\\)

The follwoing python script implements simple SGD for our movie ratings matrix. I use rank \\(r=2\\) for this example.


```python

def sgd_comp(A,r):
    np.random.seed(32)
    h = np.random.rand(A.shape[1],r) #items
    w = np.random.rand(A.shape[0],r) #users
    a,b = A.shape
    for i in range(10000):
        for j in range(a):
            for k in range(b):
                if A[j,k]==0:
                    continue
                else:
                    e = A[j,k]-np.dot(w[j,:],h[k,:].T)
                    h[k,:] = h[k,:]+0.0002*(2*e*w[j,:]-0.02*h[k,:])
                    #print(q)
                    w[j,:] = w[j,:]+0.0002*(2*e*h[k,:]-0.02*w[j,:])     
    X = np.dot(w,h.T)                
    return X,w,h

X,w,h = sgd_comp(M_obs,2)
```

The results are
$$X = \begin{bmatrix}
1.03 & \color{red}{0.16} & 3.98 & 4.98 & \color{red}{4.00} & 0.98\\
\color{red}{1.99} & 0.98 & 4.99 & \color{red}{6.09} & 4.94 & 2.02\\
4.44 & 4.43 & 1.99 & 1.61 & 1.60 & \color{red}{4.88}\\
4.51 & 4.55 & \color{red}{1.82} & 1.37 & 1.42 & 4.96
\end{bmatrix}
$$
,$$M = \begin{bmatrix}
1 & \color{red}{2} & 4 & 5 & \color{red}{4} & 1\\
\color{red}{1} & 1 & 5 & \color{red}{4} & 5 & 2\\
5 & 4 & 2 & 2 & 1 & \color{red}{5}\\
4 & 5 & 1 & \color{red}{1} & 2 & 5
\end{bmatrix}$$

The overall MSE is 0.4759. This is pretty good. We do see that for one rating we have gone above the highest possible value of 5, but overall, it does a decent job of capturing the tastes of our users. What is particularly interesting about this factorization method is observing the matrices \\(W\\) and \\(H\\)

$$W = \begin{bmatrix}
 -0.09 & 2.10\\
  0.29 & 2.55\\
  2.20 & 0.58\\
  2.27 & 0.48
\end{bmatrix},
H = \begin{bmatrix}
 1.87 & 0.57\\
 1.97 & 0.16\\
 0.40 & 1.91\\
 0.10 & 2.37\\
 0.22 & 1.91\\
 2.07 & 0.55
\end{bmatrix}$$

Observing these matrices, due to the simplicity of our original matrix, we can find an interpretation of our factorization. Each row of the \\(W\\) matrix corresponds to a user, and looking at the number, we can say that the first column is the users prefernce for Sci-Fi/Action and the second column is preference for Romantic Comedies!! We have interpretable latent variables! We can say the same for the \\(H\\) matrix, the composition of each movie as a Sci-Fi or Romantic Comedy. While not a perfect interpretation, it is useful. Unfortunately, with larger, not so obviously constructed matrices, interpretation of the latent variables is significantly more challenging, if possible at all. But this gives an idea of why latent varibles are important and the underlying idea of why a ratings matrix may be factorizable into lower rank matrices.

As mentioned, this method is non-convex. However, by alternately fixing one of the matrices \\(W\\) or \\(H\\), and performing optimization on the other, the problem becomes convex. This is the Alternating Minimization algorithm for SGD. It is a simple code modification of the above to achieve this, but not included here. There are also extensions of SGD to include bias and temporal dynamics. For more on these extensions, see *Matrix Factorization Techniques for Recommender Systems* by Koren, Bell and Volinsky.

Our last task now is to relate the matrices \\(W\\) and \\(H\\) back to SVD. As I mentioned, the optimal solution to the optimization problem is given by \\(\hat{W} = U_rS_r^{1/2}\\) and \\(\hat{H} = V_rS_r^{1/2}\\) (hats added to distinguish from the SGD matrices). When we calculate these from the orginal, filled matrix we get

$$\hat{W} = U_2S_2^{1/2}=\begin{bmatrix}
-1.79 & 1.38\\
-1.90 & 1.51\\
-2.04 & -1.28\\
-1.94 & -1.41
\end{bmatrix},
\hat{H} = V_2S_2^{1/2} =\begin{bmatrix}
-1.47 & -1.17\\
-1.58 & -1.01\\
-1.54 & 1.17\\
-1.53 & 1.15\\
-1.53 & 1.15\\
-1.73 & -1.16
\end{bmatrix}$$

These don't look anything like the \\(W\\) and \\(H\\) we found from the SGD, what's the deal? This goes back to the fact that the product \\(WH^T\\) is unaffect by right multiplication of \\(\hat{W}\\) and \\(\hat{H}\\) by a \\(r\times r\\) matrix. We can find this matric, call it \\(R\\), by the following

$$ W = \hat{W}R \Rightarrow R = \hat{W}^{+}W$$   

Note that since these are not square matrices, we have to use the psuedoinverse \\(^+\\)of \\(\hat{W}\\). Solving this, we get

$$R=\begin{bmatrix}
 -0.63 & -0.72\\
 -0.72 & 0.68
\end{bmatrix}
\text{ and }
\hat{W}R = \begin{bmatrix}
 0.12 & 2.24\\
 0.09 & 2.41\\
 2.21 & 0.61\\
 2.25 & 0.45
\end{bmatrix}
$$

While not perfect, we see that \\(\hat{W}R\\) is very close to the \\(W\\) found by SGD! You can verify for yourself that \\(\hat{H}R\\) is very close to \\(H\\). Most likely the difference is due to algorithmic or rounding errors. You can also verify that \\(R\\) is orthogonal, that is \\(RR=I\\) where \\(I\\) is the identity matrix the same size as \\(R\\). What this whould also tell you is that performing SVD on the matrix \\(WH^T\\) will give you an approximation to \\(U_rS_rV_r^T\\)! Hence, why the SGD method is actually called the SVD method in a lot of literature.

And there you have it, some matrix completion techiniques, where they come from and how they are related. I have seen some posts that claim SVD and the \\(WH^T\\) factorizations are completely separate and unrelated. As you can see here, that's simply not true and they are based on the same idea of low rank approximation, for which truncated SVD is the best solution. The \\(WH^T\\) factorization is just a different way of obtaining the truncated SVD, without actually performing truncated SVD. Hope you enjoyed!
