## here comes the logic and implementation for the actual model

import numpy as np
import math
from loader import load_data

X,Y,columns = load_data("datasets/healthcare-dataset-stroke-data.csv")

# the sigmoid function --->
def sigmoidf(x):

    return 1/(1 + np.exp(-x))
# the sigmoid function--->

# THE LOGISTIC REGRESSION MODEL PART -

n,m = X.shape
B = np.zeros((m,1))

def log_likelihood(X,Y,B):

    l = 0
    i = 0

    for x_i in X:

        p_i = sigmoidf((x_i @ B).item())
        p_i = np.clip(p_i, 1e-15, 1 - 1e-15)
        l += Y[i] * math.log(p_i) + (1 - Y[i]) * math.log(1 - p_i)
        i += 1

    return l

epsilon = 1e-3

def IRLS(B):

    while True:

        B_old = np.copy(B)

        P = np.zeros((n,1))
        i = 0

        for x_i in X:
            P[i] = sigmoidf((x_i @ B).item())
            i += 1

        W = np.diag((P * (1-P)).flatten())

        # hessian
        H = X.T @ W @ X

        # gradient
        g = X.T @ (Y.reshape(n,1) - P)

        B = B_old + np.linalg.solve(H, g)

        # checking convergence ->
        if abs(np.linalg.norm(B - B_old)) < epsilon:
            return B

        B_old = np.copy(B)

B = IRLS(B)
