## logic for predicting and testing new values

import numpy as np
import math
from scipy.stats import norm
from model import X, Y, B_probit, B_logit, columns, sigmoidf, probitf, n, m

X_new_dict = {}
X_new = []

print(B_probit)
for col in columns:
    print("Enter", col, ":")
    X_new_dict[col] = float(input())
    X_new.append(X_new_dict[col])

X_new = np.array(X_new, dtype=float).reshape(1, -1)

# Add intercept
X_new_1 = np.ones((1, 1))
X_new = np.concatenate((X_new_1, X_new), axis=1)


def predict_logit(X_new):
    p = sigmoidf((X_new @ B_logit).item())

    print("\nLogit Regression")
    print("Probability of stroke:", p)

    if p >= 0.5:
        print("Stroke: 1")
    else:
        print("Stroke: 0")


def predict_probit(X_new):
    p = norm.cdf((X_new @ B_probit).item())

    print("\nProbit Regression")
    print("Probability of stroke:", p)

    if p >= 0.5:
        print("Stroke: 1")
    else:
        print("Stroke: 0")


predict_logit(X_new)
predict_probit(X_new)
