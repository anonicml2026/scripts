#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def compute_a_sequence(lambdas, N=500, k=50, n_trials=50, random_state=0):
    """
    Compute and plot sequence {a_i} for i=1..N.

    Parameters
    ----------
    lambdas : array-like of shape (M,)
        Positive sequence of weights.
    N : int
        Number of terms in {a_i} to compute.
    k : int
        Dimension of Gaussian vectors xi_i.
    n_trials : int
        Number of Monte Carlo trials to approximate expectations.
    random_state : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    lambdas = np.array(lambdas)
    M = len(lambdas)

    # Storage for Monte Carlo estimates
    E_Mii = np.zeros(M)
    E_Mij2 = np.zeros((M, M))

    for t in range(n_trials):
        # Sample Gaussian vectors xi
        X = rng.normal(size=(M, k))  # shape (M, k)

        # Build G
        G = sum(lambdas[i] * np.outer(X[i], X[i]) for i in range(M))

        # Inverse
        Ginv = np.linalg.pinv(G)  # pseudo-inverse in case of ill-conditioning

        # Compute M_{ℓ,m}
        Mmat = X @ Ginv @ X.T   # shape (M, M)

        # Accumulate expectations
        E_Mii += np.diag(Mmat)
        E_Mij2 += Mmat**2

    # Average
    E_Mii /= n_trials
    E_Mij2 /= n_trials

    # Compute a_i
    a = np.zeros(N)
    for i in range(N):
        a[i] = 1 - 2 * lambdas[i] * E_Mii[i] + np.sum(lambdas**2 * E_Mij2[i])


    return a




# In[40]:


M = 10000
N = 1000
lambdas = np.exp(-np.arange(1,M+1,1)*0.1)  # example: λ_i = 1/i
a_vals = compute_a_sequence(lambdas, N=N, k=50, n_trials=50) 


# In[28]:


# Plot
plt.figure(figsize=(6, 5))
plt.scatter(range(1, N+1), a_vals, s=5, label=r'$\lambda_i=\frac{1}{i^{4}}, k=300$')
plt.xlabel("$i$")
plt.ylabel("$\\frac{\\mu_i}{\\lambda_i}$",rotation=90, labelpad=10, va="center")
#plt.title("$\\frac{\\mu_i}{\\lambda_i}$")
plt.legend(loc="lower right", frameon=False)
plt.grid(True)
plt.savefig('lambda_mu_4_k300.png', dpi=600, bbox_inches="tight")
plt.show() 


# In[41]:


from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
k=50

# --- solve for kappa ---
def f(kappa):
    return np.sum(lambdas / (lambdas + kappa)) - k
# bracket for root
kappa = brentq(f, 1e-20, 1e5)
print("Solved kappa =", kappa)

# --- compute theoretical curve ---
b_vals = kappa**2 / (lambdas[:N] + kappa)**2

# --- linear regression ---
X = b_vals.reshape(-1)
y = a_vals
#reg = LinearRegression().fit(X, y)
#y_pred = reg.predict(X)

slope, intercept = np.polyfit(X, y, 1)

print(f"Linear regression: slope={reg.coef_[0]:.4f}, intercept={reg.intercept_:.4f}")

# --- plot scatter with regression line ---
plt.figure(figsize=(7, 6))
plt.scatter(b_vals, a_vals, s=10, label=r"data")
plt.plot(b_vals, slope*b_vals + intercept, color="red", 
         label=f"slope$={slope:.2f}$", linestyle="--")
#plt.plot(b_vals, y_pred, color="red", linewidth=2, label="Linear fit")
plt.xlabel(r"$\frac{\varkappa^2}{(\lambda_i+\varkappa)^2}$")
plt.ylabel(r"$1-2\lambda_i \cdot \mathbb{E}[M_{i,i}]+\sum_{j=1}^\infty \lambda_j^2 \cdot \mathbb{E}[M_{i,j}^2]$")
plt.title(r'$k=$'+str(k)+r', $\lambda_i=e^{-0.1i}$')
plt.legend(frameon=False)
plt.grid(True)
plt.tight_layout()
plt.savefig('k300_4.png', dpi=600)
plt.show()


# In[ ]:




