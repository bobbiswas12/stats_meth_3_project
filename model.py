
import numpy as np
import math
from loader import load_data
from scipy.stats import norm

X, Y, columns = load_data(
    "datasets/healthcare-dataset-stroke-data.csv"
)

Y = np.asarray(Y).reshape(-1, 1)

n, m = X.shape

epsilon = 1e-6
max_iter = 1000


# THE LOGISTIC REGRESSION MODEL PART

def sigmoidf(x):
    return 1 / (1 + np.exp(-x))


def log_likelihood(X, Y, B):
    Y = np.asarray(Y).reshape(-1, 1)

    P = sigmoidf(X @ B)
    P = np.clip(P, 1e-15, 1 - 1e-15)

    return np.sum(
        Y * np.log(P) + (1 - Y) * np.log(1 - P)
    )


def IRLS(X_data, Y_data, B_init=None):

    Y_data = np.asarray(Y_data).reshape(-1, 1)
    n_data, m_data = X_data.shape

    if B_init is None:
        B = np.zeros((m_data, 1))
    else:
        B = np.copy(B_init)

    for iteration in range(max_iter):

        B_old = np.copy(B)

        P = sigmoidf(X_data @ B)

        P = np.clip(P, 1e-15, 1 - 1e-15)

        W = np.diag((P * (1 - P)).flatten())

        # Hessian
        H = X_data.T @ W @ X_data

        # Gradient
        g = X_data.T @ (Y_data - P)

        B = B_old + np.linalg.solve(H, g)

        # Checking convergence
        if np.linalg.norm(B - B_old) < epsilon:
            return B

    raise RuntimeError("no convergence :(")


B_logit = IRLS(X, Y)


# THE PROBIT REGRESSION MODEL PART

def probitf(x):
    return norm.cdf(x)


def probit_pdf(x):
    return norm.pdf(x)


def probit_log_likelihood(X, Y, B):
    Y = np.asarray(Y).reshape(-1, 1)

    eta = X @ B
    P = np.clip(probitf(eta), 1e-15, 1 - 1e-15)

    return np.sum(
        Y * np.log(P) + (1 - Y) * np.log(1 - P)
    )


def PROBIT_IRLS(X_data, Y_data, B_init=None):

    Y_data = np.asarray(Y_data).reshape(-1, 1)
    n_data, m_data = X_data.shape

    if B_init is None:
        B = np.zeros((m_data, 1))
    else:
        B = np.copy(B_init)

    for iteration in range(max_iter):

        B_old = np.copy(B)

        eta = X_data @ B

        P = np.clip(
            probitf(eta),
            1e-15,
            1 - 1e-15
        )

        D = probit_pdf(eta)

        # IRLS weights
        W = np.diag(
            (D**2 / (P * (1 - P))).flatten()
        )

        # Hessian
        H = X_data.T @ W @ X_data

        # Gradient

        g = X_data.T @ (
            D * (Y_data - P) / (P * (1 - P))
        )

        B = B_old + np.linalg.solve(H, g)

        # Checking convergence
        if np.linalg.norm(B - B_old) < epsilon:
            return B

    raise RuntimeError("no convergence :( ")


B_probit = PROBIT_IRLS(X, Y)
